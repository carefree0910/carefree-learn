import os
import dill
import json
import torch

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from PIL import Image
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import walk
from cftool.misc import is_numeric
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from cftool.misc import WithRegister
from cftool.array import to_torch
from cftool.array import to_device
from cftool.types import tensor_dict_type
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

from ..types import sample_weights_type
from ..protocol import DatasetProtocol
from ..protocol import DataLoaderProtocol
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import get_ddp_info
from ..misc.toolkit import get_world_size

try:
    import lmdb
except:
    lmdb = None


data_modules: Dict[str, Type["DataModule"]] = {}
DataModuleType = TypeVar("DataModuleType", bound="DataModule", covariant=True)


class BaseResponse(NamedTuple):
    folder: str
    base: Type["DataModule"]


class DataModule(WithRegister[DataModuleType], metaclass=ABCMeta):
    d = data_modules  # type: ignore

    id_file = "id.txt"
    info_name = "info"
    data_folder = "data"
    package_folder = "data_module"

    # inherit

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        pass

    def prepare(self, sample_weights: sample_weights_type) -> None:
        pass

    def initialize(self) -> Any:
        pass

    # internal

    @abstractmethod
    def _save_data(self, data_folder: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "DataModule":
        pass

    def _save_info(self, folder: str) -> None:
        Saving.save_dict(self.info, self.info_name, folder)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        return Saving.load_dict(cls.info_name, folder)

    @classmethod
    def _get_base(cls, folder: str) -> BaseResponse:
        folder = os.path.join(folder, cls.package_folder)
        with open(os.path.join(folder, cls.id_file), "r") as f:
            base = cls.get(f.read())
        return BaseResponse(folder, base)

    # api

    def save_info(self, folder: str) -> None:
        folder = os.path.join(folder, self.package_folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, self.id_file), "w") as f:
            f.write(self.__identifier__)
        self._save_info(folder)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        folder, base = cls._get_base(folder)
        return base._load_info(folder)

    def save(self, folder: str) -> None:
        self.save_info(folder)
        data_folder = os.path.join(folder, self.package_folder, self.data_folder)
        os.makedirs(data_folder, exist_ok=True)
        self._save_data(data_folder)

    @classmethod
    def load(
        cls,
        folder: str,
        *,
        sample_weights: sample_weights_type = None,
    ) -> "DataModule":
        folder, base = cls._get_base(folder)
        info = base._load_info(folder)
        data_folder = os.path.join(folder, cls.data_folder)
        return base._load(data_folder, info, sample_weights)


data_loaders_type = Tuple[DataLoaderProtocol, Optional[DataLoaderProtocol]]


@DataModule.register("dl")  # type: ignore
class DLDataModule(DataModule, metaclass=ABCMeta):
    @abstractmethod
    def initialize(self) -> data_loaders_type:
        pass

    def get_loaders(
        self,
        *,
        sample_weights: sample_weights_type = None,
    ) -> data_loaders_type:
        self.prepare(sample_weights)
        return self.initialize()


class DLDataset(DatasetProtocol):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, item: Any) -> Any:
        return self.dataset[item]


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        *args: Any,
        use_distributed_sampler: Optional[bool] = None,
        **kwargs: Any,
    ):
        if use_distributed_sampler is None:
            use_distributed_sampler = get_ddp_info() is not None
        if use_distributed_sampler:
            if sampler is not None and not isinstance(sampler, DistributedSampler):
                raise ValueError(
                    "`sampler` should be `DistributedSampler` "
                    "when `use_distributed_sampler` is True"
                )
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        super().__init__(dataset, batch_size, shuffle, sampler, *args, **kwargs)

    def __setattr__(self, attr: str, val: Any) -> None:
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after "
                f"{self.__class__.__name__} is initialized"
            )

        super(TorchDataLoader, self).__setattr__(attr, val)


class DLLoader(DataLoaderProtocol):
    data: DLDataset
    next_batch: tensor_dict_type

    def __init__(
        self,
        loader: DataLoader,
        batch_callback: Optional[Callable[[Any], tensor_dict_type]] = None,
        *,
        sample_weights: Optional[np.ndarray] = None,
        prefetch_device: Optional[Union[int, str]] = None,
    ):
        if sample_weights is not None:
            raise ValueError(
                "in `DLLoader`, we should introduce `sample_weights` to the original "
                "Pytorch `DataLoader` (by specifying corresponding samplers)"
            )
        super().__init__(sample_weights=sample_weights)
        self.loader = loader
        self.data = loader.dataset  # type: ignore
        self.batch_callback = batch_callback
        self.sampler_backup = loader.sampler
        self._iterator: Optional[Any] = None
        # prefetch stuffs
        if prefetch_device is not None and torch.cuda.is_available():
            prefetch_device = f"cuda:{prefetch_device}"
            self.stream = torch.cuda.Stream(prefetch_device)
        else:
            if prefetch_device is not None:
                print_warning(
                    "`prefetch_device` is specified but "
                    "cuda is not available, it will have no effects"
                )
            self.stream = None
        self.device = prefetch_device or torch.device("cpu")
        self.stop_at_next_batch = False

    def __iter__(self) -> "DLLoader":
        self.stop_at_next_batch = False
        self._iterator = self.loader.__iter__()
        self.preload()
        return self

    def __next__(self) -> tensor_dict_type:
        if self.stop_at_next_batch:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)

    def preload(self) -> None:
        try:
            batch = self._iterator.__next__()  # type: ignore
        except StopIteration:
            self.stop_at_next_batch = True
            return None

        if self.batch_callback is not None:
            batch = self.batch_callback(batch)
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                batch = to_device(batch, self.device, non_blocking=True)
        batch = to_device(batch, self.device, non_blocking=True)
        self.next_batch = batch

    @property
    def batch_size(self) -> int:  # type: ignore
        return self.loader.batch_size * get_world_size()

    def copy(self) -> "DLLoader":
        stream = self.stream
        dataset = self.data.dataset
        self.__dict__.pop("stream")
        self.data.__dict__.pop("dataset")
        copied = super().copy()
        assert isinstance(copied, DLLoader)
        self.stream = copied.stream = stream
        self.data.dataset = copied.data.dataset = dataset
        return copied

    def disable_shuffle(self) -> None:
        sampler = SequentialSampler(self.data)
        self.loader.sampler = sampler
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = sampler

    def recover_shuffle(self) -> None:
        self.loader.sampler = self.sampler_backup
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = self.sampler_backup


__all__ = [
    "DataModule",
    "DLDataModule",
    "DLDataset",
    "DataLoader",
    "DLLoader",
]
