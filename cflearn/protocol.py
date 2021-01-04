import os
import json
import math
import torch
import pprint
import logging

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from functools import partial
from tqdm.autonotebook import tqdm
from cftool.ml import Metrics
from cftool.ml import ModelPattern
from cftool.misc import register_core
from cftool.misc import timing_context
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cfdata.types import np_int_type
from cfdata.tabular import data_type
from cfdata.tabular import str_data_type
from cfdata.tabular import TaskTypes
from cfdata.tabular import DataTuple
from cfdata.tabular import TabularData
from cfdata.tabular.recognizer import Recognizer

from .types import np_dict_type
from .types import tensor_dict_type
from .types import loader_batch_type
from .types import prefetch_batch_type
from .misc.toolkit import to_prob
from .misc.toolkit import is_float
from .misc.toolkit import to_numpy
from .misc.toolkit import to_torch
from .misc.toolkit import eval_context
from .misc.toolkit import LoggingMixinWithRank
from .modules.blocks import EMA


class PipelineProtocol(LoggingMixinWithRank, metaclass=ABCMeta):
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
    def transform(
        self,
        x: Union[str, data_type],
        y: data_type = None,
        **kwargs: Any,
    ) -> DataTuple:
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
    is_onnx: bool = False
    _num_siamese: int = 1
    labels_key: str = "labels"

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
    def __next__(self) -> loader_batch_type:
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
        loader.is_onnx = is_onnx
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

    def __next__(self) -> prefetch_batch_type:
        if self.stop_at_next_batch:
            raise StopIteration
        if not self.is_cpu:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch, batch_indices = self.next_batch, self.next_batch_indices
        self.preload()
        return batch, batch_indices

    def preload(self) -> None:
        try:
            sample = self.loader.__next__()
        except StopIteration:
            self.stop_at_next_batch = True
            return None
        indices_tensor: Optional[torch.Tensor]
        if not self.return_indices:
            indices_tensor = None
        else:
            sample, batch_indices = sample  # type: ignore
            indices_tensor = to_torch(batch_indices).to(torch.long)

        self.next_batch = sample  # type: ignore
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


class TrainerState:
    def __init__(self, trainer_config: Dict[str, Any]):
        # properties
        self.step = self.epoch = 0
        self.batch_size: int
        self.num_step_per_epoch: int
        self.log_disabled: bool = False
        # settings
        self.config = trainer_config
        self.min_epoch = self.config["min_epoch"]
        self.num_epoch = self.config["num_epoch"]
        self.max_epoch = self.config["max_epoch"]
        self.log_patience = self.config.setdefault("log_patience", None)
        self.max_snapshot_file = int(self.config.setdefault("max_snapshot_file", 5))
        self.min_num_sample = self.config.setdefault("min_num_sample", 3000)
        self._snapshot_start_step = self.config.setdefault("snapshot_start_step", None)
        num_step_per_snapshot = self.config.setdefault("num_step_per_snapshot", 0)
        num_snapshot_per_epoch = self.config.setdefault("num_snapshot_per_epoch", 2)
        max_step_per_snapshot = self.config.setdefault("max_step_per_snapshot", 1000)
        plateau_start = self.config.setdefault("plateau_start_snapshot", self.num_epoch)
        self._num_step_per_snapshot = int(num_step_per_snapshot)
        self.num_snapshot_per_epoch = int(num_snapshot_per_epoch)
        self.max_step_per_snapshot = int(max_step_per_snapshot)
        self.plateau_start = int(plateau_start)

    def inject_loader(self, loader: DataLoaderProtocol) -> None:
        self.batch_size = loader.batch_size
        self.num_step_per_epoch = len(loader)

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def snapshot_start_step(self) -> int:
        if self._snapshot_start_step is not None:
            return self._snapshot_start_step
        return int(math.ceil(self.min_num_sample / self.batch_size))

    @property
    def num_step_per_snapshot(self) -> int:
        if self._num_step_per_snapshot > 0:
            return self._num_step_per_snapshot
        return max(
            1,
            min(
                self.max_step_per_snapshot,
                int(self.num_step_per_epoch / self.num_snapshot_per_epoch),
            ),
        )

    @property
    def should_train(self) -> bool:
        return self.epoch < self.num_epoch

    @property
    def should_monitor(self) -> bool:
        return self.step % self.num_step_per_snapshot == 0

    @property
    def should_log_lr(self) -> bool:
        if self.log_disabled:
            return False
        denominator = min(self.num_step_per_epoch, 10)
        return self.step % denominator == 0

    @property
    def should_log_losses(self) -> bool:
        if self.log_disabled:
            return False
        if self.log_patience is not None:
            patience = self.log_patience
        else:
            patience = max(4, int(round(self.num_step_per_epoch / 50.0)))
        denominator = min(self.num_step_per_epoch, patience)
        return self.step % denominator == 0

    @property
    def should_log_artifacts(self) -> bool:
        return self.should_log_metrics_msg

    @property
    def should_log_metrics_msg(self) -> bool:
        if self.log_disabled:
            return False
        min_period = self.max_step_per_snapshot / 3
        min_period = math.ceil(min_period / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step and self.epoch > self.min_epoch

    @property
    def should_start_monitor_plateau(self) -> bool:
        return self.step >= self.plateau_start * self.num_step_per_snapshot

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch and self.epoch < self.max_epoch

    @property
    def reached_max_epoch(self) -> bool:
        return self.epoch == self.max_epoch

    @property
    def disable_logging(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, state: TrainerState):
                self.state = state
                self.disabled = state.log_disabled

            def __enter__(self) -> None:
                self.state.log_disabled = True

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.state.log_disabled = self.disabled

        return _(self)


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: tensor_dict_type
    loss_items: Dict[str, float]


class InferenceOutputs(NamedTuple):
    results: np_dict_type
    loss_items: Optional[Dict[str, float]]
    labels: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]


# Protocols below are meant to fit with and only with `Trainer`


class TrainerDataProtocol(ABC):
    task_type: TaskTypes

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def is_reg(self) -> bool:
        return self.task_type.is_reg


class ModelProtocol(nn.Module, LoggingMixinWithRank, metaclass=ABCMeta):
    timing: bool = True
    pt_prefix: str = "model_"
    scores_file: str = "scores.json"

    __identifier__: str
    data: DataProtocol
    device: torch.device
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

    def sorted_checkpoints(self, folder: str) -> List[str]:
        # better checkpoints will be placed earlier,
        #  which means `checkpoints[0]` is the best checkpoint
        scores_path = os.path.join(folder, self.scores_file)
        if not os.path.isfile(scores_path):
            return []
        with open(scores_path, "r") as f:
            scores = json.load(f)
        files = list(scores.keys())
        scores_list = [scores[file] for file in files]
        sorted_indices = np.argsort(scores_list)[::-1]
        return [files[i] for i in sorted_indices]

    def restore_checkpoint(
        self,
        folder: str,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        checkpoints = self.sorted_checkpoints(folder)
        if not checkpoints:
            self.log_msg(  # type: ignore
                f"no model file found in {folder}",
                self.warning_prefix,  # type: ignore
                msg_level=logging.WARNING,
            )
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            self.log_msg(  # type: ignore
                f"restoring from {model_file}",
                self.info_prefix,  # type: ignore
                4,
            )
            states = torch.load(model_file, map_location=self.device)
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.load_state_dict(states, strict)
            success = True
            break
        return success

    def step(
        self,
        state: TrainerState,
        batch_idx: int,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
        loader_name: str,
        engine: Optional[Any] = None,
    ) -> StepOutputs:
        with timing_context(self, "model.forward", enable=self.timing):
            args = batch, batch_idx, state, batch_indices, loader_name
            forward_results = (engine or self)(*args)
        with timing_context(self, "loss.forward", enable=self.timing):
            loss_dict = self.loss_function(
                batch_idx,
                batch,
                batch_indices,
                forward_results,
                state,
            )
        with timing_context(self, "loss.to_item", enable=self.timing):
            loss_items = {k: v.item() for k, v in loss_dict.items()}
        return StepOutputs(forward_results, loss_dict, loss_items)

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
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
        batch_idx: Optional[int] = None,
        state: Optional[TrainerState] = None,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    @abstractmethod
    def loss_function(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
        forward_results: tensor_dict_type,
        state: Optional[TrainerState],
    ) -> tensor_dict_type:
        pass


class InferenceProtocol(ABC):
    data: DataProtocol
    model: Optional[ModelProtocol]
    is_binary: bool
    binary_metric: Optional[str]
    binary_threshold: Optional[float]
    use_binary_threshold: bool
    onnx: Any = None
    use_tqdm: bool = True
    use_grad_in_predict: bool = False

    @property
    def binary_config(self) -> Dict[str, Any]:
        return {
            "binary_metric": self.binary_metric,
            "binary_threshold": self.binary_threshold,
        }

    @property
    def output_probabilities(self) -> bool:
        if self.onnx is not None:
            return self.onnx.output_probabilities
        if self.model is None:
            raise ValueError("either `onnx` or `model` should be provided")
        return self.model.output_probabilities

    @property
    def need_binary_threshold(self) -> bool:
        if not self.use_binary_threshold:
            return False
        return self.is_binary and self.binary_metric is not None

    def to_tqdm(self, loader: PrefetchLoader) -> Union[tqdm, PrefetchLoader]:
        if not self.use_tqdm:
            return loader
        return tqdm(loader, total=len(loader), leave=False, position=2)

    # API

    def get_outputs(
        self,
        loader: PrefetchLoader,
        loader_name: Optional[str],
        *,
        use_tqdm: bool = False,
        return_loss: bool = True,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> InferenceOutputs:
        labels_key = loader.loader.labels_key
        if use_tqdm:
            loader = self.to_tqdm(loader)

        def _core() -> InferenceOutputs:
            results: Dict[str, List[np.ndarray]] = {}
            loss_items: Dict[str, List[float]] = {}
            labels = []
            for i, (batch, batch_indices) in enumerate(loader):
                local_labels = batch[labels_key]
                if local_labels is not None:
                    if not isinstance(local_labels, np.ndarray):
                        local_labels = to_numpy(local_labels)
                    labels.append(local_labels)
                if self.onnx is not None:
                    local_results = self.onnx.inference(batch)
                    local_losses = None
                else:
                    assert self.model is not None
                    with eval_context(self.model, use_grad=use_grad):
                        assert not self.model.training
                        local_kwargs = shallow_copy_dict(kwargs)
                        local_kwargs["return_loss"] = return_loss
                        local_results = self.model(
                            batch,
                            i,
                            state,
                            batch_indices,
                            loader_name,
                            **local_kwargs,
                        )
                    if not return_loss:
                        local_losses = None
                    else:
                        with eval_context(self.model, use_grad=use_grad):
                            assert not self.model.training
                            local_losses = self.model.loss_function(
                                i,
                                batch,
                                batch_indices,
                                local_results,
                                state,
                            )
                for k, v in local_results.items():
                    if v is None:
                        continue
                    if self.onnx is not None:
                        v_np = v
                    else:
                        v_np = to_numpy(v)
                    results.setdefault(k, []).append(v_np)
                if local_losses is not None:
                    for k, v in local_losses.items():
                        loss_items.setdefault(k, []).append(v.item())

            return InferenceOutputs(
                {k: np.vstack(v) for k, v in results.items()},
                None
                if not loss_items
                else {k: sum(v) / len(v) for k, v in loss_items.items()},
                None if not labels else np.vstack(labels),
                None,
            )

        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        try:
            return _core()
        except:
            use_grad = self.use_grad_in_predict = True
            return _core()

    def predict_from_outputs(
        self,
        outputs: InferenceOutputs,
        return_all: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, np_dict_type]:
        results = outputs.results
        # regression
        if self.data.is_reg:
            return_key = kwargs.get("return_key", "predictions")
            fn = partial(self.data.recover_labels, inplace=True)
            if not return_all:
                predictions = results[return_key]
                if requires_recover:
                    if predictions.shape[1] == 1:
                        return fn(predictions)
                    return np.apply_along_axis(fn, axis=0, arr=predictions).squeeze()
                return predictions
            if not requires_recover:
                return results
            recovered = {}
            for k, v in results.items():
                if is_float(v):
                    if v.shape[1] == 1:
                        v = fn(v)
                    else:
                        v = np.apply_along_axis(fn, axis=0, arr=v).squeeze()
                recovered[k] = v
            return recovered

        # classification
        def _return(new_predictions: np.ndarray) -> Union[np.ndarray, np_dict_type]:
            if not return_all:
                return new_predictions
            results["predictions"] = new_predictions
            return results

        predictions = results["logits"] = results["predictions"]
        if returns_probabilities:
            if not self.output_probabilities:
                predictions = to_prob(predictions)
            return _return(predictions)
        if not self.is_binary or self.binary_threshold is None:
            return _return(predictions.argmax(1).reshape([-1, 1]))

        if self.output_probabilities:
            probabilities = predictions
        else:
            probabilities = to_prob(predictions)
        return _return(self.predict_with(probabilities))

    def predict(
        self,
        loader: PrefetchLoader,
        *,
        return_all: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        loader_name: Optional[str] = None,
        return_loss: bool = False,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, np_dict_type]:
        # Notice : when `return_all` is True,
        #  there might not be `predictions` key in the results
        outputs = self.get_outputs(
            loader,
            loader_name,
            use_tqdm=use_tqdm,
            return_loss=return_loss,
            **shallow_copy_dict(kwargs),
        )
        return self.predict_from_outputs(
            outputs,
            return_all,
            requires_recover,
            returns_probabilities,
            **shallow_copy_dict(kwargs),
        )

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
        loader: PrefetchLoader,
        loader_name: str,
        *,
        return_loss: bool = True,
        use_tqdm: bool = False,
    ) -> Optional[InferenceOutputs]:
        if not self.need_binary_threshold:
            return None
        outputs = self.get_outputs(
            loader,
            loader_name,
            return_loss=return_loss,
            getting_metrics=True,
            use_tqdm=use_tqdm,
        )
        results = self.predict_from_outputs(
            outputs,
            return_all=True,
            returns_probabilities=True,
        )
        labels = outputs.labels
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

        new_outputs = InferenceOutputs(*outputs[:3], probabilities)
        return None if loader_name == "tr" else new_outputs


__all__ = [
    "PipelineProtocol",
    "PatternPipeline",
    "DataSplit",
    "DataProtocol",
    "SamplerProtocol",
    "DataLoaderProtocol",
    "PrefetchLoader",
    "TrainerState",
    "StepOutputs",
    "InferenceOutputs",
    "TrainerDataProtocol",
    "ModelProtocol",
    "InferenceProtocol",
]
