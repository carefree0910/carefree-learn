import os
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type
from torch.utils.data import Dataset

from .core import DLLoader
from .core import DLDataset
from .core import DataLoader
from .core import DLDataModule
from ..types import sample_weights_type
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import ORIGINAL_LABEL_KEY


class TensorDataset(Dataset):
    def __init__(
        self,
        x: Tensor,
        y: Optional[Tensor],
        others: Optional[tensor_dict_type] = None,
    ):
        self.x = x
        self.y = y
        self.others = others

    def __getitem__(self, index: int) -> tensor_dict_type:
        item = {INPUT_KEY: self.x[index]}
        if self.y is not None:
            label = self.y[index]
            item.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        if self.others is not None:
            for k, v in self.others.items():
                item[k] = v[index]
        return item

    def __len__(self) -> int:
        return self.x.shape[0]


class TensorDictDataset(Dataset):
    def __init__(
        self,
        x: tensor_dict_type,
        y: Optional[Tensor],
    ):
        self.x = x
        self.y = y
        self.x_keys = sorted(self.x)

    def __getitem__(self, index: int) -> tensor_dict_type:
        item = {k: self.x[k][index] for k in self.x_keys}
        if self.y is not None:
            label = self.y[index]
            item.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        return item

    def __len__(self) -> int:
        return self.x[self.x_keys[0]].shape[0]


# api


@DLDataModule.register("tensor")
class TensorData(DLDataModule):
    files = [
        "x_train.pt",
        "y_train.pt",
        "x_valid.pt",
        "y_valid.pt",
        "train_others.pt",
        "valid_others.pt",
    ]

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
        def _get_data(x: Any, y: Any, others: Any) -> DLDataset:
            return DLDataset(TensorDataset(x, y, others))

        self.train_data = _get_data(self.x_train, self.y_train, self.train_others)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid, self.valid_others)

    def initialize(self) -> Tuple[DLLoader, Optional[DLLoader]]:
        train_loader = DLLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = DLLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader

    def _save_data(self, data_folder: str) -> None:
        all_data = [self.x_train, self.y_train, self.x_valid, self.y_valid]
        all_data += [self.train_others, self.valid_others]
        for data, file in zip(all_data, self.files):
            if data is not None:
                torch.save(data, os.path.join(data_folder, file))

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "TensorData":
        args = []
        for file in cls.files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else torch.load(path))
        data = cls(*args, **info)
        data.prepare(sample_weights)
        return data


@DLDataModule.register("tensor_dict")
class TensorDictData(DLDataModule):
    files = ["x_train.pt", "y_train.pt", "x_valid.pt", "y_valid.pt"]

    def __init__(
        self,
        x_train: tensor_dict_type,
        y_train: Optional[Tensor] = None,
        x_valid: Optional[tensor_dict_type] = None,
        y_valid: Optional[Tensor] = None,
        *,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        def _get_data(x: Any, y: Any) -> DLDataset:
            return DLDataset(TensorDictDataset(x, y))

        self.train_data = _get_data(self.x_train, self.y_train)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid)

    def initialize(self) -> Tuple[DLLoader, Optional[DLLoader]]:
        train_loader = DLLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = DLLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader

    def _save_data(self, data_folder: str) -> None:
        all_data = [self.x_train, self.y_train, self.x_valid, self.y_valid]
        for data, file in zip(all_data, self.files):
            if data is not None:
                torch.save(data, os.path.join(data_folder, file))

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "TensorData":
        args = []
        for file in cls.files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else torch.load(path))
        data = cls(*args, **info)  # type: ignore
        data.prepare(sample_weights)
        return data


@DLDataModule.register("dummy")
class DummyData(TensorData):
    def __init__(
        self,
        *,
        num_samples: int = 1,
        batch_size: int = 1,
        use_valid: bool = False,
    ):
        dummy = torch.zeros([num_samples, 1])
        x_valid = y_valid = dummy if use_valid else None
        super().__init__(dummy, dummy, x_valid, y_valid, batch_size=batch_size)


__all__ = [
    "TensorData",
    "TensorDictData",
    "DummyData",
]
