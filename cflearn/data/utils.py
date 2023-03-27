import torch

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from cftool.array import arr_type
from cftool.array import to_device

from ..schema import IDataset
from ..schema import IDataLoader
from ..schema import DataProcessor
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import BATCH_INDICES_KEY
from ..constants import ORIGINAL_LABEL_KEY
from ..misc.toolkit import np_batch_to_tensor
from ..misc.toolkit import tensor_batch_to_np


TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]
TArrayDict = Union[np_dict_type, tensor_dict_type]


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


class IArrayDataset(IDataset, metaclass=ABCMeta):
    def __init__(
        self,
        x: arr_type,
        y: Optional[arr_type],
        processor: DataProcessor,
        others: Optional[TArrayDict] = None,
    ):
        self.x = x
        self.y = y
        self.processor = processor
        self.others = others

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        batch = {INPUT_KEY: self.x[item]}
        if self.y is not None:
            label = self.y[item]
            batch.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        if self.others is not None:
            for k, v in self.others.items():
                batch[k] = v[item]
        batch = self.processor.postprocess_item(batch)
        batch = tensor_batch_to_np(batch)
        return batch

    def __len__(self) -> int:
        return len(self.x)

    def to_npd(self) -> np_dict_type:
        tensors = dict(x=self.x)
        if self.y is not None:
            tensors["y"] = self.y
        if self.others is not None:
            tensors.update(self.others)
        return tensor_batch_to_np(tensors)

    def from_npd(self, npd: np_dict_type) -> None:
        d = self.before_load(npd)
        self.x = d.pop("x")
        self.y = d.pop("y", None)
        self.others = d

    @abstractmethod
    def before_load(self, npd: np_dict_type) -> TArrayDict:
        pass


class IArrayDictDataset(IDataset, metaclass=ABCMeta):
    def __init__(
        self,
        x: TArrayDict,
        y: Optional[arr_type],
        processor: DataProcessor,
    ):
        self.x = x
        self.y = y
        self.processor = processor
        self.x_keys = sorted(self.x)

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> TArrayDict:
        batch = {k: self.x[k][item] for k in self.x_keys}
        if self.y is not None:
            label = self.y[item]
            batch.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        batch = self.processor.postprocess_item(batch)
        return batch

    def __len__(self) -> int:
        return len(self.x[self.x_keys[0]])

    def to_npd(self) -> np_dict_type:
        tensors = self.x
        if self.y is not None:
            tensors["y"] = self.y
        return tensor_batch_to_np(tensors)

    def from_npd(self, npd: np_dict_type) -> None:
        d = self.before_load(npd)
        self.y = d.pop("y", None)
        self.x = d
        self.x_keys = sorted(self.x)

    @abstractmethod
    def before_load(self, npd: np_dict_type) -> TArrayDict:
        pass


class IArrayLoader(IDataLoader):
    dataset: IDataset

    cursor: int
    indices: np.ndarray

    shuffle: bool
    shuffle_backup: bool
    sample_weights: Optional[np.ndarray]

    def __init__(
        self,
        dataset: IDataset,
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

    def __iter__(self) -> "IArrayLoader":
        self.cursor = 0
        self.indices = get_weighted_indices(len(self.dataset), self.sample_weights)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Union[np_dict_type, tensor_dict_type]:
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

    def copy(self) -> "IArrayLoader":
        return self.__class__(
            self.dataset,
            self.batch_size,
            shuffle=self.shuffle,
            sample_weights=self.sample_weights,
        )


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


__all__ = [
    "get_weighted_indices",
    "IArrayLoader",
    "TensorBatcher",
]
