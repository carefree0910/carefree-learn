import torch

import numpy as np

from typing import Tuple
from typing import Union
from typing import Optional
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from cftool.array import to_device

from ..schema import IDataset
from ..schema import IDataLoader
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import np_batch_to_tensor


TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


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
