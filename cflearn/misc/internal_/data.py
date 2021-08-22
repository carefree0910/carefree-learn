import numpy as np
import torch.distributed as dist

from typing import Any
from typing import Tuple
from typing import Callable
from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler

from ...types import tensor_dict_type
from ...types import sample_weights_type
from ...protocol import DataProtocol
from ...protocol import DataLoaderProtocol
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import BATCH_INDICES_KEY
from ...misc.toolkit import to_torch


class DataModule:
    def prepare(self, sample_weights: sample_weights_type) -> None:
        pass

    def initialize(self) -> Any:
        pass


class DLDataModule(DataModule):
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


@DataProtocol.register("ml")
class MLData(DataProtocol):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)


@DataLoaderProtocol.register("ml")
class MLLoader(DataLoaderProtocol):
    data: MLData
    cursor: int
    indices: np.ndarray

    def __init__(
        self,
        data: MLData,
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


@DataProtocol.register("cv")
class CVData(DataProtocol):
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
            use_distributed_sampler = dist.is_initialized()
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
    data: CVData

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
    def batch_size(self):
        batch_size = self.loader.batch_size
        if dist.is_initialized():
            batch_size *= dist.get_world_size()
        return batch_size

    def copy(self) -> "CVLoader":
        if not hasattr(self.data.dataset, "lmdb"):
            copied = super().copy()
            assert isinstance(copied, CVLoader)
        else:
            lmdb, context = self.data.dataset.lmdb, self.data.dataset.context
            self.data.dataset.__dict__.pop("lmdb")
            self.data.dataset.__dict__.pop("context")
            copied = super().copy()
            assert isinstance(copied, CVLoader)
            copied.data.dataset.lmdb, copied.data.dataset.context = lmdb, context
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
    "MLData",
    "MLLoader",
    "CVData",
    "CVLoader",
    "DataLoader",
    "get_weighted_indices",
]
