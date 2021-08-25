import os

import numpy as np
import torch.distributed as dist

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Callable
from typing import Optional
from cftool.misc import Saving
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

from ...types import tensor_dict_type
from ...types import sample_weights_type
from ...protocol import DatasetProtocol
from ...protocol import DataLoaderProtocol
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import BATCH_INDICES_KEY
from ...misc.toolkit import to_torch
from ...misc.toolkit import get_ddp_info
from ...misc.toolkit import WithRegister


data_modules: Dict[str, Type["DataModule"]] = {}


class DataModule(WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DataModule"]] = data_modules

    id_file = "id.txt"
    info_name = "info"
    package_folder = "data_module"

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        pass

    def prepare(self, sample_weights: sample_weights_type) -> None:
        pass

    def initialize(self) -> Any:
        pass

    def save(self, folder: str) -> None:
        folder = os.path.join(folder, self.package_folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, self.id_file), "w") as f:
            f.write(self.__identifier__)
        self.save_info(folder)

    def save_info(self, folder: str) -> None:
        Saving.save_dict(self.info, self.info_name, folder)

    @classmethod
    def load(cls, folder: str) -> Dict[str, Any]:
        folder = os.path.join(folder, cls.package_folder)
        with open(os.path.join(folder, cls.id_file), "r") as f:
            base = cls.get(f.read())
        return base.load_info(folder)

    @classmethod
    def load_info(cls, folder: str) -> Dict[str, Any]:
        return Saving.load_dict(cls.info_name, folder)


@DataModule.register("dl")
class DLDataModule(DataModule, metaclass=ABCMeta):
    train_loader: DataLoaderProtocol
    valid_loader: Optional[DataLoaderProtocol]

    def initialize(self) -> Tuple[DataLoaderProtocol, Optional[DataLoaderProtocol]]:
        pass


def get_weighted_indices(n: int, weights: Optional[np.ndarray]) -> np.ndarray:
    indices = np.arange(n)
    if weights is not None:
        numbers = np.random.multinomial(n, weights)
        indices = indices.repeat(numbers)
    return indices


@DatasetProtocol.register("ml")
class MLDataset(DatasetProtocol):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)


@DataLoaderProtocol.register("ml")
class MLLoader(DataLoaderProtocol):
    data: MLDataset
    cursor: int
    indices: np.ndarray

    def __init__(
        self,
        data: MLDataset,
        shuffle: bool,
        *,
        name: Optional[str] = None,
        batch_size: int = 128,
        sample_weights: Optional[np.ndarray] = None,
    ):
        if sample_weights is not None and len(data) != len(sample_weights):
            raise ValueError(
                f"the number of data samples ({len(data)}) is not identical with "
                f"the number of sample weights ({len(sample_weights)})"
            )
        super().__init__(sample_weights=sample_weights)
        self.data = data
        self.shuffle = shuffle
        self.shuffle_backup = shuffle
        self.name = name
        self.batch_size = batch_size

    def __iter__(self) -> "MLLoader":
        self.cursor = 0
        self.indices = get_weighted_indices(len(self.data), self.sample_weights)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> tensor_dict_type:
        start = self.cursor
        if start >= len(self.data):
            raise StopIteration
        self.cursor += self.batch_size
        indices = self.indices[start : self.cursor]
        return {
            INPUT_KEY: to_torch(self.data.x[indices]),
            LABEL_KEY: None if self.data.y is None else to_torch(self.data.y[indices]),
            BATCH_INDICES_KEY: to_torch(indices),
        }

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "MLLoader":
        return MLLoader(
            self.data,
            self.shuffle,
            name=self.name,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights,
        )


@DatasetProtocol.register("cv")
class CVDataset(DatasetProtocol):
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


@DataLoaderProtocol.register("cv")
class CVLoader(DataLoaderProtocol):
    data: CVDataset

    def __init__(
        self,
        loader: DataLoader,
        batch_callback: Optional[Callable[[Any], tensor_dict_type]] = None,
        *,
        sample_weights: Optional[np.ndarray] = None,
    ):
        if sample_weights is not None:
            raise ValueError(
                "in `CVLoader`, we should introduce `sample_weights` to the original "
                "Pytorch `DataLoader` (by specifying corresponding samplers)"
            )
        super().__init__(sample_weights=sample_weights)
        self.loader = loader
        self.data = loader.dataset  # type: ignore
        self.batch_callback = batch_callback
        self.sampler_backup = loader.sampler
        self._iterator: Optional[Any] = None

    def __iter__(self) -> "CVLoader":
        self._iterator = self.loader.__iter__()
        return self

    def __next__(self) -> tensor_dict_type:
        batch = self._iterator.__next__()  # type: ignore
        if self.batch_callback is None:
            return batch
        return self.batch_callback(batch)

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def batch_size(self) -> int:  # type: ignore
        batch_size = self.loader.batch_size
        if dist.is_initialized():
            batch_size *= dist.get_world_size()
        return batch_size

    def copy(self) -> "CVLoader":
        dataset = self.data.dataset
        self.data.__dict__.pop("dataset")
        copied = super().copy()
        assert isinstance(copied, CVLoader)
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
    "MLDataset",
    "MLLoader",
    "CVDataset",
    "CVLoader",
    "DataLoader",
    "get_weighted_indices",
]
