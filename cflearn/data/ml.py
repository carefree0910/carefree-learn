import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import Saving
from cftool.misc import WithRegister
from cftool.array import to_torch
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .core import DLDataModule
from ..types import sample_weights_type
from ..protocol import DatasetProtocol
from ..protocol import DataLoaderProtocol
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import BATCH_INDICES_KEY


ml_loader_callbacks: Dict[str, Type["IMLLoaderCallback"]] = {}
ml_data_modifiers: Dict[str, Type["IMLDataModifier"]] = {}


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


class IMLDataset(DatasetProtocol, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        pass


class IMLLoaderCallback(WithRegister["IMLLoaderCallback"], metaclass=ABCMeta):
    d = ml_loader_callbacks

    def __init__(self, loader: "IMLLoader"):
        self.loader = loader

    @property
    def data(self) -> IMLDataset:
        return self.loader.data

    # changes may happen inplace, but the final batch will be returned anyway
    @abstractmethod
    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        pass


class IMLLoader(DataLoaderProtocol, metaclass=ABCMeta):
    callback: str

    cursor: int
    indices: np.ndarray

    data: IMLDataset
    shuffle: bool
    shuffle_backup: bool
    name: Optional[str]
    batch_size: int
    use_numpy: bool

    def __init__(
        self,
        data: IMLDataset,
        shuffle: bool,
        *,
        name: Optional[str] = None,
        batch_size: int = 128,
        sample_weights: Optional[np.ndarray] = None,
        use_numpy: bool = False,
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
        self.use_numpy = use_numpy

    def __iter__(self) -> "IMLLoader":
        self.cursor = 0
        self.indices = get_weighted_indices(len(self.data), self.sample_weights)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Union[np_dict_type, tensor_dict_type]:
        start = self.cursor
        if start >= len(self.data):
            raise StopIteration
        self.cursor += self.batch_size
        indices = self.indices[start : self.cursor]
        batch = self.data[indices]
        batch = self._make_callback().process_batch(batch)
        batch.setdefault(BATCH_INDICES_KEY, indices)
        if self.use_numpy:
            return batch
        return {k: None if v is None else to_torch(v) for k, v in batch.items()}

    def _make_callback(self) -> IMLLoaderCallback:
        return IMLLoaderCallback.make(self.callback, {"loader": self})

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "IMLLoader":
        return self.__class__(
            self.data,
            self.shuffle,
            name=self.name,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights,
        )


class IMLDataInfo(NamedTuple):
    """
    * input_dim (int) : final dim that the model will receive
    * num_history (int) : number of history, useful in time series tasks
    * num_classes (int | None) : number of classes
      -> will be used as `output_dim` if `is_classification` is True & `output_dim` is not specified
    * is_classification (bool | None) : whether current task is a classification task
      -> it should always be provided unless it's at inference time
    """

    input_dim: int
    num_history: int
    num_classes: Optional[int] = None
    is_classification: Optional[bool] = None


class IMLDataModifier(WithRegister["IMLDataModifier"], metaclass=ABCMeta):
    d = ml_data_modifiers

    def __init__(self, data: "IMLData"):
        self.data = data

    # abstract

    @abstractmethod
    def save(self, data_folder: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, data_folder: str, info: Dict[str, Any]) -> "IMLData":
        pass

    # optional callbacks

    # changes should happen inplace
    def permute_info_in_save(self, info: Dict[str, Any]) -> None:
        pass

    # changes should happen inplace
    @classmethod
    def permute_info_in_load(
        cls,
        data_cls: Type["IMLData"],
        info: Dict[str, Any],
    ) -> None:
        pass


class IMLData(DLDataModule, metaclass=ABCMeta):
    modifier: str

    @abstractmethod
    def get_info(self) -> IMLDataInfo:
        pass

    @property
    def info(self) -> Dict[str, Any]:
        return self.get_info()._asdict()

    def _make_modifier(self) -> IMLDataModifier:
        return IMLDataModifier.make(self.modifier, {"data": self})

    def _save_info(self, folder: str) -> None:
        info = self.info
        self._make_modifier().permute_info_in_save(info)
        Saving.save_dict(info, self.info_name, folder)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        info = super()._load_info(folder)
        IMLDataModifier.get(cls.modifier).permute_info_in_load(cls, info)
        return info

    def _save_data(self, data_folder: str) -> None:
        self._make_modifier().save(data_folder)

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "IMLData":
        data = IMLDataModifier.get(cls.modifier).load(data_folder, info)
        data.loaded = True
        data.prepare(sample_weights)
        return data


def register_ml_loader_callback(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable:
    return IMLLoaderCallback.register(name, allow_duplicate=allow_duplicate)


def register_ml_data_modifier(name: str, *, allow_duplicate: bool = False) -> Callable:
    return IMLDataModifier.register(name, allow_duplicate=allow_duplicate)


@DatasetProtocol.register("ml")
class MLDataset(IMLDataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray], **others: np.ndarray):
        super().__init__()
        self.x = x
        self.y = y
        self.others = others

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        batch = {
            INPUT_KEY: self.x[item],
            LABEL_KEY: None if self.y is None else self.y[item],
        }
        for k, v in self.others.items():
            batch[k] = v[item]
        return batch

    def __len__(self) -> int:
        return len(self.x)


@IMLLoaderCallback.register("basic")
class BasicMLLoaderCallback(IMLLoaderCallback):
    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        return batch


@DataLoaderProtocol.register("ml")
class MLLoader(IMLLoader):
    callback = "basic"


__all__ = [
    "get_weighted_indices",
    "register_ml_loader_callback",
    "register_ml_data_modifier",
    "IMLDataset",
    "IMLLoaderCallback",
    "IMLLoader",
    "IMLDataInfo",
    "IMLDataModifier",
    "IMLData",
    "MLDataset",
    "MLLoader",
]
