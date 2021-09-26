import os
import dill
import json
import lmdb

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
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import is_numeric
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

from ..types import tensor_dict_type
from ..types import sample_weights_type
from ..protocol import DatasetProtocol
from ..protocol import DataLoaderProtocol
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import walk
from ..misc.toolkit import to_torch
from ..misc.toolkit import get_ddp_info
from ..misc.toolkit import get_world_size
from ..misc.toolkit import WithRegister


data_modules: Dict[str, Type["DataModule"]] = {}


class DataModule(WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DataModule"]] = data_modules

    id_file = "id.txt"
    info_name = "info"
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

    def _save_info(self, folder: str) -> None:
        Saving.save_dict(self.info, self.info_name, folder)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        return Saving.load_dict(cls.info_name, folder)

    # api

    def save(self, folder: str) -> None:
        folder = os.path.join(folder, self.package_folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, self.id_file), "w") as f:
            f.write(self.__identifier__)
        self._save_info(folder)

    @classmethod
    def load(cls, folder: str) -> Dict[str, Any]:
        folder = os.path.join(folder, cls.package_folder)
        with open(os.path.join(folder, cls.id_file), "r") as f:
            base = cls.get(f.read())
        return base._load_info(folder)


@DataModule.register("dl")
class DLDataModule(DataModule, metaclass=ABCMeta):
    test_transform: Optional[Callable]
    transform_file = "transform.pkl"

    @abstractmethod
    def initialize(self) -> Tuple[DataLoaderProtocol, Optional[DataLoaderProtocol]]:
        pass

    def _save_info(self, folder: str) -> None:
        super()._save_info(folder)
        with open(os.path.join(folder, self.transform_file), "wb") as f:
            dill.dump(self.test_transform, f)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        info = super()._load_info(folder)
        transform_path = os.path.join(folder, cls.transform_file)
        if not os.path.isfile(transform_path):
            test_transform = None
        else:
            with open(transform_path, "rb") as f:
                test_transform = dill.load(f)
        info["test_transform"] = test_transform
        return info


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


class DLDataset(DatasetProtocol):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, item: Any) -> Any:
        return self.dataset[item]


@DatasetProtocol.register("cv")
class CVDataset(DLDataset):
    pass


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

    def __init__(
        self,
        loader: DataLoader,
        batch_callback: Optional[Callable[[Any], tensor_dict_type]] = None,
        *,
        sample_weights: Optional[np.ndarray] = None,
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

    def __iter__(self) -> "DLLoader":
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
        return self.loader.batch_size * get_world_size()

    def copy(self) -> "DLLoader":
        dataset = self.data.dataset
        self.data.__dict__.pop("dataset")
        copied = super().copy()
        assert isinstance(copied, DLLoader)
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


@DataLoaderProtocol.register("cv")
class CVLoader(DLLoader):
    data: CVDataset


cf_transforms: Dict[str, Type["Transforms"]] = {}


class Transforms(WithRegister):
    d: Dict[str, Type["Transforms"]] = cf_transforms

    fn: Any

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, inp: Any, *args: Any, **kwargs: Any) -> Any:
        if self.need_batch_process and not isinstance(inp, dict):
            raise ValueError(f"`inp` should be a batch for {self.__class__.__name__}")
        return self.fn(inp, *args, **kwargs)

    @property
    @abstractmethod
    def need_batch_process(self) -> bool:
        pass

    @property
    def need_numpy(self) -> bool:
        return self.need_batch_process

    @classmethod
    def convert(
        cls,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
    ) -> Optional["Transforms"]:
        if transform is None:
            return None
        if isinstance(transform, Transforms):
            return transform
        if callable(transform):
            need_batch = (transform_config or {}).get("need_batch_process", False)
            return Function(transform, need_batch)
        if transform_config is None:
            transform_config = {}
        if isinstance(transform, str):
            return cls.make(transform, transform_config)
        transform_list = [cls.make(t, transform_config.get(t, {})) for t in transform]
        return Compose(transform_list)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "Transforms":
        split = name.split("_")
        if len(split) >= 3 and split[-2] == "with":
            name = "_".join(split[:-2])
            config.setdefault("label_alias", split[-1])
        return super().make(name, config)


class Function(Transforms):
    def __init__(self, fn: Callable, need_batch_process: bool = False):
        super().__init__()
        self.fn = fn
        self._need_batch_process = need_batch_process

    @property
    def need_batch_process(self) -> bool:
        return self._need_batch_process


@Transforms.register("compose")
class Compose(Transforms):
    def __init__(self, transform_list: List[Transforms]):
        super().__init__()
        if len(set(t.need_batch_process for t in transform_list)) > 1:
            raise ValueError(
                "all transforms should have identical "
                "`need_batch_process` property in `Compose`"
            )
        self.fn = transforms.Compose(transform_list)
        self.transform_list = transform_list

    @property
    def need_batch_process(self) -> bool:
        return self.transform_list[0].need_batch_process

    @property
    def need_numpy(self) -> bool:
        return self.transform_list[0].need_numpy


def _default_lmdb_path(folder: str, split: str) -> str:
    return f"{folder}.{split}"


class LMDBItem(NamedTuple):
    image: np.ndarray
    labels: Dict[str, Any]


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        folder: str,
        split: str,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        extra_label_names: Optional[List[str]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = os.path.abspath(os.path.join(folder, split))
        support_appendix = {".jpg", ".png", ".npy"}
        self.img_paths = list(
            map(
                lambda file: os.path.join(self.folder, file),
                filter(
                    lambda file: file[-4:] in support_appendix,
                    os.listdir(self.folder),
                ),
            )
        )
        if lmdb_config is None:
            self.lmdb = self.context = None
            with open(os.path.join(self.folder, f"{LABEL_KEY}.json"), "r") as f:
                self.labels = {LABEL_KEY: json.load(f)}
            if extra_label_names is not None:
                for name in extra_label_names:
                    el_file = f"{name}_{LABEL_KEY}.json"
                    with open(os.path.join(self.folder, el_file), "r") as f:
                        self.labels[name] = json.load(f)
            self.length = len(self.img_paths)
        else:
            self.lmdb_config = shallow_copy_dict(lmdb_config)
            self.lmdb_config.setdefault("path", _default_lmdb_path(folder, split))
            self.lmdb_config.setdefault("lock", False)
            self.lmdb_config.setdefault("meminit", False)
            self.lmdb_config.setdefault("readonly", True)
            self.lmdb_config.setdefault("readahead", False)
            self.lmdb_config.setdefault("max_readers", 32)
            self.lmdb = lmdb.open(**self.lmdb_config)
            self.context = self.lmdb.begin(buffers=True, write=False)
            with self.lmdb.begin(write=False) as context:
                self.length = int(context.get("length".encode("ascii")).decode("ascii"))
        self.transform = Transforms.convert(transform, transform_config)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.context is not None:
            key = str(index).encode("ascii")
            item: LMDBItem = dill.loads(self.context.get(key))
            img = Image.fromarray(item.image)
            labels = item.labels
            for k, v in labels.items():
                if is_numeric(v):
                    labels[k] = np.array([v])
        else:
            file = self.img_paths[index]
            img = Image.open(file)
            labels = {k: v[file] for k, v in self.labels.items()}
            for k, v in labels.items():
                if is_numeric(v):
                    v = np.array([v])
                elif isinstance(v, str) and v.endswith(".npy"):
                    v = np.load(v)
                labels[k] = v
        if self.transform is None:
            img = to_torch(np.array(img).astype(np.float32))
            for k, v in labels.items():
                if isinstance(v, np.ndarray):
                    labels[k] = to_torch(v)
            sample = {INPUT_KEY: img}
            sample.update(labels)
            return sample
        if not self.transform.need_numpy:
            sample = {INPUT_KEY: img}
        else:
            sample = {INPUT_KEY: np.array(img).astype(np.float32) / 255.0}
        if self.transform.need_batch_process:
            sample.update(labels)
            return self.transform(sample)
        for k, v in labels.items():
            if isinstance(v, np.ndarray):
                labels[k] = to_torch(v)
        sample = {INPUT_KEY: self.transform(sample[INPUT_KEY])}
        sample.update(labels)
        return sample

    def __len__(self) -> int:
        return self.length


class InferenceImageFolderDataset(Dataset):
    def __init__(self, folder: str, transform: Optional[Callable]):
        self.folder = os.path.abspath(folder)
        self.img_paths: List[str] = []
        walk(
            self.folder,
            lambda _, path: self.img_paths.append(path),
            {".jpg", ".png"},
        )
        self.transform = transform

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        img = Image.open(self.img_paths[index])
        if self.transform is None:
            img = to_torch(np.array(img).astype(np.float32))
            return {INPUT_KEY: img}
        return {INPUT_KEY: self.transform(img)}

    def __len__(self) -> int:
        return len(self.img_paths)

    def make_loader(self, batch_size: int, num_workers: int = 0) -> CVLoader:
        return CVLoader(DataLoader(self, batch_size, num_workers=num_workers))


__all__ = [
    "DataModule",
    "DLDataModule",
    "MLDataset",
    "MLLoader",
    "DLDataset",
    "CVDataset",
    "DataLoader",
    "DLLoader",
    "CVLoader",
    "Transforms",
    "Function",
    "Compose",
    "get_weighted_indices",
]
