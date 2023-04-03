import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.array import to_numpy
from cftool.types import np_dict_type
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler

from ...schema import IData
from ...schema import IDataset
from ...schema import DataConfig
from ...schema import IDataLoader
from ...schema import DataProcessor
from ...schema import DataProcessorConfig
from ...misc.toolkit import get_ddp_info
from ...misc.toolkit import get_world_size


class TorchDataset(IDataset):
    def __init__(self, dataset: IDataset, processor: DataProcessor) -> None:
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        batch = self.dataset[item]
        batch = self.processor.postprocess_item(batch)
        return batch


class _DataLoader(DataLoader):
    def __init__(
        self,
        dataset: TorchDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler[int]] = None,
        *,
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
        super().__init__(dataset, batch_size, shuffle, sampler, **kwargs)

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

        super(DataLoader, self).__setattr__(attr, val)


class TorchDataLoader(IDataLoader):
    dataset: TorchDataset

    def __init__(
        self,
        loader: _DataLoader,
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
        self.use_numpy = False
        self.dataset = loader.dataset  # type: ignore
        self.sampler_backup = loader.sampler
        self._iterator: Optional[_BaseDataLoaderIter] = None

    def __iter__(self) -> "TorchDataLoader":
        self._iterator = self.loader.__iter__()
        return self

    def __next__(self) -> np_dict_type:
        if self._iterator is None:
            raise StopIteration
        batch = self._iterator.__next__()
        batch = {
            k: to_numpy(v) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def batch_size(self) -> int:  # type: ignore
        return self.loader.batch_size * get_world_size()

    def copy(self) -> "TorchDataLoader":
        dataset = self.dataset
        self.__dict__.pop("dataset")
        copied = super().copy()
        assert isinstance(copied, TorchDataLoader)
        self.dataset = copied.dataset = dataset
        return copied

    def disable_shuffle(self) -> None:
        sampler = SequentialSampler(self.dataset)
        self.loader.sampler = sampler
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = sampler

    def recover_shuffle(self) -> None:
        self.loader.sampler = self.sampler_backup
        if hasattr(self.loader, "batch_sampler"):
            self.loader.batch_sampler.sampler = self.sampler_backup


@DataConfig.register("torch")
class TorchDataConfig(DataConfig):
    use_distributed_sampler: Optional[bool] = None


@IData.register("torch")
class TorchData(IData):
    """
    A thin wrapper for general Datasets, utilizing the `DataLoader` system from pytorch.

    Examples
    --------
    >>> train_dataset = ...
    >>> valid_dataset = None / ...
    >>> data = cflearn.TorchData.init(config, processor_config)
    >>> data.fit(train_dataset, x_valid=valid_dataset)

    or

    >>> train_dataset = ...
    >>> valid_dataset = None / ...
    >>> data = cflearn.TorchData.build(
    >>>     train_dataset,
    >>>     valid_dataset,
    >>>     config=config,
    >>>     processor_config=processor_config,
    >>> )

    """

    config: TorchDataConfig
    train_dataset: IDataset
    valid_dataset: Optional[IDataset]

    # inheritance

    @property
    def config_base(self) -> Type[TorchDataConfig]:
        return TorchDataConfig

    @property
    def train_kw(self) -> Dict[str, Any]:
        return dict(
            dataset=self.train_dataset,
            sample_weights=self.train_weights,
            shuffle=self.config.shuffle_train,
            batch_size=self.config.batch_size,
            use_distributed_sampler=self.config.use_distributed_sampler,
        )

    @property
    def valid_kw(self) -> Dict[str, Any]:
        return dict(
            dataset=self.valid_dataset,
            sample_weights=self.valid_weights,
            shuffle=self.config.shuffle_valid,
            batch_size=self.config.valid_batch_size or self.config.batch_size,
            use_distributed_sampler=self.config.use_distributed_sampler,
        )

    def get_loaders(self) -> Tuple[TorchDataLoader, Optional[TorchDataLoader]]:
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
        self.train_dataset = TorchDataset(self.bundle.x_train, self.processor)  # type: ignore
        train_loader = self._make_loader(self.train_kw)
        if self.bundle.x_valid is None:
            self.valid_dataset = None
            valid_loader = None
        else:
            self.valid_dataset = TorchDataset(self.bundle.x_valid, self.processor)  # type: ignore
            valid_loader = self._make_loader(self.valid_kw)
        return train_loader, valid_loader  # type: ignore

    def _make_loader(self, kw: Dict[str, Any]) -> Optional[TorchDataLoader]:
        dataset = kw.pop("dataset")
        if dataset is None:
            return None
        sw = kw.pop("sample_weights")
        return TorchDataLoader(_DataLoader(dataset, **kw), sample_weights=sw)

    # api

    @classmethod
    def build(
        cls,
        x_train: Union[IDataset, Any],
        x_valid: Optional[Union[IDataset, Any]] = None,
        *,
        config: Optional[TorchDataConfig] = None,
        processor_config: Optional[DataProcessorConfig] = None,
    ) -> "TorchData":
        self: TorchData = cls.init(config, processor_config)
        self.fit(x_train, x_valid=x_valid)
        return self


__all__ = [
    "TorchDataset",
    "TorchDataLoader",
    "TorchDataConfig",
    "TorchData",
]
