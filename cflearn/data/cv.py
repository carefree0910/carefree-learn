import os
import dill
import json

import numpy as np

from abc import abstractmethod
from PIL import Image
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import walk
from cftool.misc import is_numeric
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister
from cftool.array import to_torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from .core import DLLoader
from .core import DLDataset
from .core import DataLoader
from ..protocol import IDataset
from ..protocol import IDataLoader
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY

try:
    import lmdb
except:
    lmdb = None


@IDataset.register("cv")
class CVDataset(DLDataset):
    pass


@IDataLoader.register("cv")
class CVLoader(DLLoader):
    data: CVDataset


cf_transforms: Dict[str, Type["Transforms"]] = {}


class Transforms(WithRegister["Transforms"]):
    d = cf_transforms

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


def default_lmdb_path(folder: str, split: str) -> str:
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
        if lmdb_config is None or lmdb is None:
            if lmdb_config is not None:
                print_warning(
                    "`lmdb` is not installed, so `lmdb_config` will be ignored"
                )
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
            self.lmdb_config.setdefault("path", default_lmdb_path(folder, split))
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

    def make_loader(
        self,
        batch_size: int,
        num_workers: int = 0,
        prefetch_device: Optional[Union[int, str]] = None,
    ) -> CVLoader:
        return CVLoader(
            DataLoader(self, batch_size, num_workers=num_workers),
            prefetch_device=prefetch_device,
        )


__all__ = [
    "LMDBItem",
    "CVDataset",
    "ImageFolderDataset",
    "InferenceImageFolderDataset",
    "CVLoader",
    "Transforms",
    "Function",
    "Compose",
    "default_lmdb_path",
]
