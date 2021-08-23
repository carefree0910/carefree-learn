import json

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
from .core import TensorDataset
from .core import ImageFolderDataset
from .transforms import Transforms
from ....types import tensor_dict_type
from ....types import sample_weights_type
from ....misc.internal_ import CVDataset
from ....misc.internal_ import CVLoader
from ....misc.internal_ import DataLoader
from ....misc.internal_ import DLDataModule


@DLDataModule.register("mnist")
class MNISTData(DLDataModule):
    def __init__(
        self,
        *,
        root: str = "data",
        shuffle: bool = True,
        batch_size: int = 64,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
        label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]] = None,
    ):
        self.root = root
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transform = Transforms.convert(transform, transform_config)
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


@DLDataModule.register("tensor")
class TensorData(DLDataModule):
    def __init__(
        self,
        x_train: Tensor,
        y_train: Optional[Tensor] = None,
        x_valid: Optional[Tensor] = None,
        y_valid: Optional[Tensor] = None,
        train_others: Optional[tensor_dict_type] = None,
        valid_others: Optional[tensor_dict_type] = None,
        *,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.train_others = train_others
        self.valid_others = valid_others
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        def _get_data(x: Any, y: Any, others: Any) -> CVDataset:
            return CVDataset(TensorDataset(x, y, others))

        self.train_data = _get_data(self.x_train, self.y_train, self.train_others)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid, self.valid_others)

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        train_loader = CVLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = CVLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader


@DLDataModule.register("image_folder")
class ImageFolderData(DLDataModule):
    def __init__(
        self,
        folder: str,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        transform: Optional[Union[str, Transforms]] = None,
        test_shuffle: Optional[bool] = None,
        test_transform: Optional[Union[str, Transforms]] = None,
        lmdb_configs: Optional[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.shuffle = shuffle
        self.transform = transform
        self.test_shuffle = test_shuffle
        self.test_transform = test_transform
        self.lmdb_configs = lmdb_configs
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        d = shallow_copy_dict(self.kw)
        d["test_shuffle"] = self.test_shuffle
        try:
            json.dumps(self.lmdb_configs)
            d["lmdb_configs"] = self.lmdb_configs
        except Exception as err:
            d["lmdb_configs"] = str(err)
        return d

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "train",
                self.transform,
                lmdb_configs=self.lmdb_configs,
            )
        )
        self.valid_data = CVDataset(
            ImageFolderDataset(
                self.folder,
                "valid",
                self.test_transform or self.transform,
                lmdb_configs=self.lmdb_configs,
            )
        )

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        d = shallow_copy_dict(self.kw)
        train_loader = CVLoader(DataLoader(self.train_data, **d))  # type: ignore
        d["shuffle"] = self.test_shuffle or self.shuffle
        valid_loader = CVLoader(DataLoader(self.valid_data, **d))  # type: ignore
        return train_loader, valid_loader


__all__ = [
    "MNISTData",
    "TensorData",
    "ImageFolderData",
]
