import torch
import pprint

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from typing import Any
from typing import Set
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.ml import Metrics
from cftool.ml import ModelPattern
from cftool.misc import register_core
from cftool.misc import LoggingMixin
from cfdata.types import np_int_type
from cfdata.types import np_float_type
from cfdata.tabular import data_type
from cfdata.tabular import batch_type
from cfdata.tabular import str_data_type
from cfdata.tabular import TaskTypes
from cfdata.tabular import DataTuple
from cfdata.tabular import TabularData
from cfdata.tabular.recognizer import Recognizer

from .types import np_dict_type
from .types import tensor_dict_type
from .types import tensor_batch_type
from .misc.toolkit import to_torch
from .modules.blocks import EMA


class PipelineProtocol(LoggingMixin, metaclass=ABCMeta):
    def __init__(self) -> None:
        self.data = TabularData.simple("reg", simplify=True)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ) -> "PipelineProtocol":
        self.data.read(x, y)
        return self._core(x, y, x_cv, y_cv)

    @abstractmethod
    def _core(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ) -> "PipelineProtocol":
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def to_pattern(self, **predict_config: Any) -> ModelPattern:
        return ModelPattern(predict_method=lambda x: self.predict(x, **predict_config))


class PatternPipeline(PipelineProtocol):
    def __init__(self, pattern: ModelPattern):
        super().__init__()
        self.pattern = pattern

    def _core(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ) -> "PipelineProtocol":
        pass

    def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.pattern.predict(x, **kwargs)

    def to_pattern(self, **predict_config: Any) -> ModelPattern:
        return self.pattern


data_dict: Dict[str, Type["DataProtocol"]] = {}
sampler_dict: Dict[str, Type["SamplerProtocol"]] = {}
loader_dict: Dict[str, Type["DataLoaderProtocol"]] = {}


class DataSplit(NamedTuple):
    split: "DataProtocol"
    remained: "DataProtocol"
    split_indices: np.ndarray
    remained_indices: np.ndarray


class DataProtocol(ABC):
    is_ts: bool
    is_clf: bool
    is_simplify: bool
    raw_dim: int
    num_classes: int

    raw: DataTuple
    converted: DataTuple
    processed: DataTuple
    ts_indices: Set[int]
    recognizers: Dict[int, Optional[Recognizer]]

    _verbose_level: int
    _has_column_names: bool

    @abstractmethod
    def __init__(self, **kwargs: Any):
        self._verbose_level = kwargs.get("verbose_level", 2)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def read_file(
        self,
        file_path: str,
        *,
        contains_labels: bool = True,
    ) -> Tuple[str_data_type, Optional[str_data_type]]:
        pass

    @abstractmethod
    def read(
        self,
        x: Union[str, data_type],
        y: Optional[Union[int, data_type]] = None,
        *,
        contains_labels: bool = True,
        **kwargs: Any,
    ) -> "DataProtocol":
        pass

    @abstractmethod
    def split(self, n: Union[int, float], *, order: str = "auto") -> DataSplit:
        pass

    @abstractmethod
    def split_with_indices(
        self,
        split_indices: np.ndarray,
        remained_indices: np.ndarray,
    ) -> DataSplit:
        pass

    @abstractmethod
    def transform(self, x: Union[str, data_type], y: data_type = None) -> DataTuple:
        pass

    @abstractmethod
    def transform_labels(self, y: data_type) -> np.ndarray:
        pass

    @abstractmethod
    def recover_labels(self, y: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def copy_to(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        *,
        contains_labels: bool = True,
    ) -> "DataProtocol":
        pass

    @abstractmethod
    def save(
        self,
        folder: str,
        *,
        compress: bool = True,
        retain_data: bool = True,
        remove_original: bool = True,
    ) -> "DataProtocol":
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        folder: str,
        *,
        compress: bool = True,
        verbose_level: int = 0,
    ) -> "DataProtocol":
        pass

    @property
    def is_reg(self) -> bool:
        return not self.is_clf

    @property
    def task_type(self) -> TaskTypes:
        if not self.is_ts:
            if self.is_clf:
                return TaskTypes.CLASSIFICATION
            return TaskTypes.REGRESSION
        if self.is_clf:
            return TaskTypes.TIME_SERIES_CLF
        return TaskTypes.TIME_SERIES_REG

    @classmethod
    def get(cls, name: str) -> Type["DataProtocol"]:
        return data_dict[name]

    @classmethod
    def make(cls, name: str, **kwargs: Any) -> "DataProtocol":
        return cls.get(name)(**kwargs)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global data_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, data_dict, before_register=before)


class SamplerProtocol(ABC):
    shuffle: bool

    @abstractmethod
    def __init__(self, data: DataProtocol, **kwargs: Any):
        self.data = data

    @classmethod
    def get(cls, name: str) -> Type["SamplerProtocol"]:
        return sampler_dict[name]

    @classmethod
    def make(cls, name: str, data: DataProtocol, **kwargs: Any) -> "SamplerProtocol":
        return cls.get(name)(data, **kwargs)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global sampler_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, sampler_dict, before_register=before)


class DataLoaderProtocol(ABC):
    _num_siamese: int = 1

    data: DataProtocol
    sampler: SamplerProtocol
    enabled_sampling: bool
    return_indices: bool
    batch_size: int
    _verbose_level: int

    @abstractmethod
    def __init__(
        self,
        batch_size: int,
        sampler: SamplerProtocol,
        *,
        return_indices: bool = False,
        verbose_level: int = 2,
        **kwargs: Any,
    ):
        self.batch_size = batch_size
        self.sampler = sampler
        self.return_indices = return_indices
        self._verbose_level = verbose_level

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> "DataLoaderProtocol":
        pass

    @abstractmethod
    def __next__(self) -> batch_type:
        pass

    @abstractmethod
    def copy(self) -> "DataLoaderProtocol":
        pass

    @classmethod
    def get(cls, name: str) -> Type["DataLoaderProtocol"]:
        return loader_dict[name]

    @classmethod
    def make(cls, name: str, *args: Any, **kwargs: Any) -> "DataLoaderProtocol":
        return cls.get(name)(*args, **kwargs)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global loader_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, loader_dict, before_register=before)


class PrefetchLoader:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        device: Union[str, torch.device],
        *,
        is_onnx: bool = False,
    ):
        self.loader = loader
        self.device = device
        self.is_onnx = is_onnx
        self.data = loader.data
        self.return_indices = loader.return_indices
        self.stream = None if self.is_cpu else torch.cuda.Stream(device)
        self.next_batch: Union[np_dict_type, tensor_dict_type]
        self.next_batch_indices: Optional[torch.Tensor]
        self.stop_at_next_batch = False
        self.batch_size = loader.batch_size
        self._num_siamese = loader._num_siamese

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "PrefetchLoader":
        self.stop_at_next_batch = False
        self.loader.__iter__()
        self.preload()
        return self

    def __next__(self) -> tensor_batch_type:
        if self.stop_at_next_batch:
            raise StopIteration
        if not self.is_cpu:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch, batch_indices = self.next_batch, self.next_batch_indices
        self.preload()
        return batch, batch_indices

    def preload(self) -> None:
        try:
            sample = next(self.loader)
        except StopIteration:
            self.stop_at_next_batch = True
            return None
        indices_tensor: Optional[torch.Tensor]
        if not self.return_indices:
            x_batch, y_batch = sample
            indices_tensor = None
        else:
            (x_batch, y_batch), batch_indices = sample
            indices_tensor = to_torch(batch_indices).to(torch.long)
        self.next_batch = self._collate_batch(x_batch, y_batch)

        if self.is_cpu:
            self.next_batch_indices = indices_tensor
            return None

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: None if v is None else v.to(self.device, non_blocking=True)
                for k, v in self.next_batch.items()
            }
            if indices_tensor is None:
                self.next_batch_indices = None
            else:
                indices_tensor = indices_tensor.to(self.device, non_blocking=True)
                self.next_batch_indices = indices_tensor

    @property
    def is_cpu(self) -> bool:
        if self.is_onnx:
            return True
        if isinstance(self.device, str):
            return self.device == "cpu"
        return self.device.type == "cpu"

    def _collate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Union[np_dict_type, tensor_dict_type]:
        x_batch = x_batch.astype(np_float_type)
        if self.is_onnx:
            if y_batch is None:
                y_batch = np.zeros([*x_batch.shape[:-1], 1], np_int_type)
            arrays = [x_batch, y_batch]
        else:
            x_batch = to_torch(x_batch)
            if y_batch is not None:
                y_batch = to_torch(y_batch)
                if self.data.is_clf:
                    y_batch = y_batch.to(torch.long)
            arrays = [x_batch, y_batch]
        return dict(zip(["x_batch", "y_batch"], arrays))


# This protocol is meant to fit with and only with `Trainer`
class ModelProtocol(nn.Module, LoggingMixin, metaclass=ABCMeta):
    __identifier__: str
    ema: Optional[EMA] = None
    num_train: Optional[int] = None
    num_valid: Optional[int] = None

    @property
    def use_ema(self) -> bool:
        return self.ema is not None

    def init_ema(self) -> None:
        if self.config is None:
            return None
        ema_decay = self.config.setdefault("ema_decay", 0.0)
        if 0.0 < ema_decay < 1.0:
            named_params = list(self.named_parameters())
            self.ema = EMA(ema_decay, named_params)  # type: ignore

    def apply_ema(self) -> None:
        if self.ema is None:
            raise ValueError("`ema` is not defined")
        self.ema()

    def info(self, *, return_only: bool = False) -> str:
        msg = "\n".join(["=" * 100, "configurations", "-" * 100, ""])
        msg += (
            pprint.pformat(self.configurations, compact=True) + "\n" + "-" * 100 + "\n"
        )
        msg += "\n".join(["=" * 100, "parameters", "-" * 100, ""])
        for name, param in self.named_parameters():
            if param.requires_grad:
                msg += name + "\n"
        msg += "\n".join(["-" * 100, "=" * 100, "buffers", "-" * 100, ""])
        for name, param in self.named_buffers():
            msg += name + "\n"
        msg += "\n".join(
            ["-" * 100, "=" * 100, "structure", "-" * 100, str(self), "-" * 100, ""]
        )
        if not return_only:
            self.log_block_msg(msg, verbose_level=4)  # type: ignore
        all_msg, msg = msg, "=" * 100 + "\n"
        msg += f"{self.info_prefix}training data : {self.num_train or 'unknown'}\n"
        msg += f"{self.info_prefix}valid    data : {self.num_valid or 'unknown'}\n"
        msg += "-" * 100
        if not return_only:
            self.log_block_msg(msg, verbose_level=3)  # type: ignore
        return "\n".join([all_msg, msg])

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskTypes:
        pass

    @property
    @abstractmethod
    def configurations(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def output_probabilities(self) -> bool:
        pass

    @abstractmethod
    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    @abstractmethod
    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
        forward_results: tensor_dict_type,
        batch_step: int,
    ) -> tensor_dict_type:
        pass


# This protocol is meant to fit with and only with `Trainer`
class InferenceProtocol(ABC):
    is_binary: bool
    binary_metric: Optional[str]
    binary_threshold: Optional[float]
    use_binary_threshold: bool

    @abstractmethod
    def predict(
        self,
        loader: PrefetchLoader,
        *,
        return_all: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        loader_name: Optional[str] = None,
        batch_step: int = -1,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, np_dict_type]:
        pass

    @property
    def binary_config(self) -> Dict[str, Any]:
        return {
            "binary_metric": self.binary_metric,
            "binary_threshold": self.binary_threshold,
        }

    @property
    def need_binary_threshold(self) -> bool:
        if not self.use_binary_threshold:
            return False
        return self.is_binary and self.binary_metric is not None

    def predict_with(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.is_binary or self.binary_threshold is None:
            return probabilities.argmax(1).reshape([-1, 1])
        predictions = (
            (probabilities[..., 1] >= self.binary_threshold)
            .astype(np_int_type)
            .reshape([-1, 1])
        )
        return predictions

    def generate_binary_threshold(
        self,
        loader: Optional[PrefetchLoader] = None,
        loader_name: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.need_binary_threshold:
            return None
        if loader is None:
            raise ValueError("`loader` should be provided")
        results = self.predict(
            loader,
            return_all=True,
            returns_probabilities=True,
            loader_name=loader_name,
        )
        labels = results["labels"]
        probabilities = results["predictions"]
        try:
            threshold = Metrics.get_binary_threshold(
                labels,
                probabilities,
                self.binary_metric,
            )
            self.binary_threshold = threshold.item()
        except ValueError:
            self.binary_threshold = None

        if loader_name == "tr":
            return None
        return labels, probabilities


__all__ = [
    "PipelineProtocol",
    "PatternPipeline",
    "DataSplit",
    "DataProtocol",
    "SamplerProtocol",
    "DataLoaderProtocol",
    "PrefetchLoader",
    "ModelProtocol",
    "InferenceProtocol",
]
