import torch

import numpy as np

from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict
from cftool.array import to_device
from cftool.types import arr_type
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from ..schema import IDataset
from ..schema import IDLModel
from ..schema import DataArgs
from ..schema import DataBundle
from ..schema import DataConfig
from ..schema import IDataLoader
from ..schema import DataProcessor
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import eval_context
from ..misc.toolkit import get_device
from ..misc.toolkit import np_batch_to_tensor
from ..misc.toolkit import tensor_batch_to_np


TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]
TArrayDict = Union[np_dict_type, tensor_dict_type]
TArrayDataset = Union["IArrayDataset", "IArrayDictDataset"]


def get_weighted_indices(
    n: int,
    weights: Optional[np.ndarray],
    ensure_all_occur: bool = False,
) -> np.ndarray:
    indices = np.arange(n)
    if weights is not None:
        numbers = np.random.multinomial(n, weights)
        if ensure_all_occur:
            numbers += 1
        indices = indices.repeat(numbers)
    return indices


class IArrayDataset(IDataset):
    x: arr_type
    y: Optional[arr_type]
    processor: DataProcessor
    others: Optional[TArrayDict] = None


class IArrayDictDataset(IDataset):
    x: TArrayDict
    y: Optional[arr_type]
    processor: DataProcessor
    x_keys: List[str]


class ArrayLoader(IDataLoader):
    dataset: TArrayDataset

    cursor: int
    indices: np.ndarray

    shuffle: bool
    shuffle_backup: bool
    sample_weights: Optional[np.ndarray]

    def __init__(
        self,
        dataset: TArrayDataset,
        batch_size: int = 128,
        *,
        shuffle: bool = False,
        sample_weights: Optional[np.ndarray] = None,
    ):
        if sample_weights is not None and len(dataset) != len(sample_weights):
            raise ValueError(
                f"the number of data samples ({len(dataset)}) is not identical with "
                f"the number of sample weights ({len(sample_weights)})"
            )
        super().__init__(sample_weights=sample_weights)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_backup = shuffle
        self.sample_weights = sample_weights

    def __iter__(self) -> "ArrayLoader":
        self.cursor = 0
        self.indices = get_weighted_indices(len(self.dataset), self.sample_weights)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> TArrayDict:
        start = self.cursor
        if start >= len(self.dataset):
            raise StopIteration
        self.cursor += self.batch_size
        indices = self.indices[start : self.cursor]
        batch = self.dataset[indices]
        batch.setdefault(BATCH_INDICES_KEY, indices)
        return batch

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "ArrayLoader":
        return self.__class__(
            self.dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sample_weights=self.sample_weights,
        )


class IArrayDataMixin(ABC):
    config: DataConfig
    bundle: DataBundle
    processor: DataProcessor
    train_dataset: TArrayDataset
    valid_dataset: Optional[TArrayDataset]
    train_weights: Optional[np.ndarray]
    valid_weights: Optional[np.ndarray]

    @property
    def train_kw(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            sample_weights=self.train_weights,
        )

    @property
    def valid_kw(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.config.valid_batch_size or self.config.batch_size,
            shuffle=self.config.shuffle_valid,
            sample_weights=self.valid_weights,
        )

    def fit(
        self: "IArrayDataMixin",
        x_train: arr_type,
        y_train: Optional[arr_type] = None,
        x_valid: Optional[arr_type] = None,
        y_valid: Optional[arr_type] = None,
        train_others: Optional[TArrayDict] = None,
        valid_others: Optional[TArrayDict] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "IArrayDataMixin":
        if train_others is not None:
            train_others = tensor_batch_to_np(train_others)
        if valid_others is not None:
            valid_others = tensor_batch_to_np(valid_others)
        return super().fit(  # type: ignore
            x_train,
            y_train,
            x_valid,
            y_valid,
            train_others,
            valid_others,
            *args,
            **kwargs,
        )

    def get_loaders(self) -> Tuple[ArrayLoader, Optional[ArrayLoader]]:
        if not self.processor.is_ready:
            raise ValueError(
                "`processor` should be ready before calling `initialize`, "
                "did you forget to call the `prepare` method first?"
            )
        if self.bundle is None:
            raise ValueError(
                "`bundle` property is not initialized, "
                "did you forget to call the `fit` method first?"
            )
        self.train_dataset = self.get_dataset(self.bundle.train_args)
        train_loader = ArrayLoader(self.train_dataset, **self.train_kw)
        if self.bundle.x_valid is None:
            valid_loader = None
        else:
            self.valid_dataset = self.get_dataset(self.bundle.valid_args)
            valid_loader = ArrayLoader(self.valid_dataset, **self.valid_kw)
        return train_loader, valid_loader

    @abstractmethod
    def get_dataset(self, data_args: DataArgs) -> TArrayDataset:
        pass


def predict_array_data(
    m: IDLModel,
    data: IArrayDataMixin,
    *,
    batch_size: Optional[int] = None,
    **predict_kwargs: Any,
) -> Any:
    if batch_size is not None:
        data.config.batch_size = batch_size
    loader = data.get_loaders()[0]
    results = []
    with eval_context(m):
        tensor_batcher = TensorBatcher(loader, get_device(m))
        for i, batch in enumerate(tensor_batcher):
            batch = shallow_copy_dict(batch)
            results.append(m.run(i, batch, **predict_kwargs))
    final = {}
    for k in results[0]:
        final[k] = torch.cat([rs[k] for rs in results], dim=0)
    return final


class TensorBatcher:
    def __init__(self, loader: IDataLoader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "TensorBatcher":
        self.loader.__iter__()
        return self

    def __next__(self) -> tensor_dict_type:
        npd = self.loader.__next__()
        batch = np_batch_to_tensor(npd)
        return to_device(batch, self.device)

    def to(self, device: torch.device) -> None:
        self.device = device

    def get_full_batch(self) -> tensor_dict_type:
        return np_batch_to_tensor(self.loader.get_full_batch())


__all__ = [
    "get_weighted_indices",
    "IArrayDataset",
    "IArrayDictDataset",
    "ArrayLoader",
    "IArrayDataMixin",
    "predict_array_data",
    "TensorBatcher",
]
