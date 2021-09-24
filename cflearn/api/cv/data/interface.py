import json

from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from functools import partial
from cftool.misc import shallow_copy_dict
from torchvision.datasets import MNIST

from .core import batch_callback
from .core import ImageFolderDataset
from .core import InferenceImageFolderDataset
from .transforms import Transforms
from ....types import sample_weights_type
from ....constants import DATA_CACHE_DIR
from ....misc.internal_ import CVDataset
from ....misc.internal_ import CVLoader
from ....misc.internal_ import DataLoader
from ....misc.internal_ import DLDataModule


class CVDataModule(DLDataModule, metaclass=ABCMeta):
    test_transform: Optional[Transforms]


@DLDataModule.register("mnist")
class MNISTData(CVDataModule):
    def __init__(
        self,
        *,
        root: str = DATA_CACHE_DIR,
        shuffle: bool = True,
        batch_size: int = 64,
        transform: Optional[Union[str, List[str], Transforms, Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]] = None,
    ):
        self.root = root
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform = Transforms.convert(transform, transform_config)
        self.test_transform = self.transform
        self.label_callback = label_callback

    @property
    def info(self) -> Dict[str, Any]:
        return dict(root=self.root, shuffle=self.shuffle, batch_size=self.batch_size)

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            MNIST(
                self.root,
                transform=self.transform,
                download=True,
            )
        )
        self.valid_data = CVDataset(
            MNIST(
                self.root,
                train=False,
                transform=self.transform,
                download=True,
            )
        )

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        train_loader = CVLoader(
            DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
            partial(batch_callback, self.label_callback),
        )
        valid_loader = CVLoader(
            DataLoader(
                self.valid_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
            partial(batch_callback, self.label_callback),
        )
        return train_loader, valid_loader


@DLDataModule.register("image_folder")
class ImageFolderData(CVDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        extra_label_names: Optional[List[str]] = None,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
        test_shuffle: Optional[bool] = None,
        test_transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        test_transform_config: Optional[Dict[str, Any]] = None,
        lmdb_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.shuffle = shuffle
        self.extra_label_names = extra_label_names
        self.transform = Transforms.convert(transform, transform_config)
        self.test_shuffle = test_shuffle
        if test_transform is None:
            test_transform = transform
            if test_transform_config is None:
                test_transform_config = transform_config
        self.test_transform = Transforms.convert(test_transform, test_transform_config)
        if self.test_transform is None:
            self.test_transform = self.transform
        self.lmdb_config = lmdb_config
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        d = shallow_copy_dict(self.kw)
        d["test_shuffle"] = self.test_shuffle
        try:
            json.dumps(self.lmdb_config)
            d["lmdb_config"] = self.lmdb_config
        except Exception as err:
            d["lmdb_config"] = str(err)
        return d

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "train",
                self.transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )
        self.valid_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "valid",
                self.test_transform,
                extra_label_names=self.extra_label_names,
                lmdb_config=self.lmdb_config,
            )
        )

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        d = shallow_copy_dict(self.kw)
        train_loader = CVLoader(DataLoader(self.train_data, **d))  # type: ignore
        d["shuffle"] = self.test_shuffle or self.shuffle
        valid_loader = CVLoader(DataLoader(self.valid_data, **d))  # type: ignore
        return train_loader, valid_loader


class InferenceImageFolderData(CVDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        transform: Optional[Union[str, List[str], Transforms, Callable]] = None,
        transform_config: Optional[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = Transforms.convert(transform, transform_config)
        self.kw: Dict[str, Any] = dict(batch_size=batch_size, num_workers=num_workers)
        self.prepare(None)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.dataset = InferenceImageFolderDataset(self.folder, self.transform)

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        loader = self.dataset.make_loader(self.batch_size, self.num_workers)
        return loader, None

    @classmethod
    def from_package(
        cls,
        src_folder: str,
        *,
        package_folder: str,
        batch_size: int,
        num_workers: int = 0,
    ) -> "InferenceImageFolderData":
        info = cls.load(package_folder)
        transform = info["test_transform"]
        return cls(
            src_folder,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
        )


__all__ = [
    "CVDataModule",
    "MNISTData",
    "ImageFolderData",
    "InferenceImageFolderData",
]
