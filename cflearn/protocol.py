import math
import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from copy import deepcopy
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from onnxruntime import InferenceSession
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler

from .constants import *
from .types import losses_type
from .types import np_dict_type
from .types import tensor_dict_type
from .types import sample_weights_type
from .misc.toolkit import to_numpy
from .misc.toolkit import to_device
from .misc.toolkit import to_standard
from .misc.toolkit import eval_context


data_dict: Dict[str, Type["DataProtocol"]] = {}
loader_dict: Dict[str, Type["DataLoaderProtocol"]] = {}
model_dict: Dict[str, Type["ModelProtocol"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["LossProtocol"]] = {}
metric_dict: Dict[str, Type["MetricProtocol"]] = {}


T = TypeVar("T")


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def make_multiple(
        cls,
        names: Union[str, List[str]],
        configs: Optional[Dict[str, Any]] = None,
    ) -> Union[T, List[T]]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            return cls.get(names)(**configs)  # type: ignore
        return [
            cls.get(name)(**shallow_copy_dict(configs.get(name, {})))  # type: ignore
            for name in names
        ]

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


# data


class DataProtocol(ABC, WithRegister):
    d: Dict[str, Type["DataProtocol"]] = data_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataLoaderProtocol(ABC, WithRegister):
    d: Dict[str, Type["DataLoaderProtocol"]] = loader_dict
    data: DataProtocol
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "DataLoaderProtocol":
        pass

    @abstractmethod
    def __next__(self) -> tensor_dict_type:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def copy(self) -> "DataLoaderProtocol":
        return deepcopy(self)


# model


class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["ModelProtocol"]] = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return list(self.parameters())[0].device

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self.forward(batch_idx, batch)


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: tensor_dict_type


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]


class InferenceOutputs(NamedTuple):
    forward_results: np_dict_type
    labels: Optional[np.ndarray]
    metric_outputs: Optional[MetricsOutputs]
    loss_items: Optional[Dict[str, float]]


class ModelWithCustomSteps(ModelProtocol, metaclass=ABCMeta):
    custom_train_step: bool = True
    custom_evaluate_step: bool = True

    @abstractmethod
    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        pass

    @abstractmethod
    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        pass


# trainer


class TrainerState:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        *,
        num_epoch: int,
        max_epoch: int,
        extension: int = 5,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 5,
        num_snapshot_per_epoch: int = 2,
        num_step_per_log: int = 350,
        num_step_per_snapshot: Optional[int] = None,
        max_step_per_snapshot: int = 2000,
    ):
        self.step = self.epoch = 0
        self.batch_size = loader.batch_size
        self.num_step_per_epoch = len(loader)
        self.num_epoch = num_epoch
        self.max_epoch = max_epoch
        self.extension = extension
        self.enable_logging = enable_logging
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_step_per_log = num_step_per_log
        if num_step_per_snapshot is None:
            num_step_per_snapshot = max(1, int(len(loader) / num_snapshot_per_epoch))
            num_step_per_snapshot = min(max_step_per_snapshot, num_step_per_snapshot)
        self.num_step_per_snapshot = num_step_per_snapshot

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def should_train(self) -> bool:
        return self.epoch < self.num_epoch

    @property
    def should_monitor(self) -> bool:
        return self.step % self.num_step_per_snapshot == 0

    @property
    def should_log_lr(self) -> bool:
        if not self.enable_logging:
            return False
        denominator = min(self.num_step_per_epoch, 10)
        return self.step % denominator == 0

    @property
    def should_log_losses(self) -> bool:
        if not self.enable_logging:
            return False
        patience = max(4, int(round(self.num_step_per_epoch / 50.0)))
        denominator = min(self.num_step_per_epoch, patience)
        return self.step % denominator == 0

    @property
    def should_log_artifacts(self) -> bool:
        return self.should_log_metrics_msg

    @property
    def should_log_metrics_msg(self) -> bool:
        if not self.enable_logging:
            return False
        if self.is_terminate:
            return True
        min_period = math.ceil(self.num_step_per_log / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch and self.epoch < self.max_epoch

    @property
    def reached_max_epoch(self) -> bool:
        return self.epoch > self.max_epoch

    @property
    def disable_logging(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, state: TrainerState):
                self.state = state
                self.enabled = state.enable_logging

            def __enter__(self) -> None:
                self.state.enable_logging = False

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.state.enable_logging = self.enabled

        return _(self)


class TrainerMonitor(ABC, WithRegister):
    d: Dict[str, Type["TrainerMonitor"]] = monitor_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def snapshot(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def check_terminate(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def punish_extension(self) -> None:
        pass

    def handle_extension(self, state: TrainerState) -> None:
        if state.should_extend_epoch:
            self.punish_extension()
            new_epoch = state.num_epoch + state.extension
            state.num_epoch = min(new_epoch, state.max_epoch)


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    metric_outputs: Optional[MetricsOutputs]


# loss


class LossProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["LossProtocol"]] = loss_dict

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.config = config or {}
        self._init_config()
        self._reduction = reduction

    def _init_config(self) -> None:
        pass

    def _reduce(self, losses: torch.Tensor) -> torch.Tensor:
        if self._reduction == "none":
            return losses
        if self._reduction == "mean":
            return losses.mean()
        if self._reduction == "sum":
            return losses.sum()
        raise NotImplementedError(f"reduction '{self._reduction}' is not implemented")

    @abstractmethod
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        # return losses without reduction
        pass

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        losses = self._core(forward_results, batch, state)
        if isinstance(losses, torch.Tensor):
            return {LOSS_KEY: self._reduce(losses)}
        # requires returns a value with LOSS_KEY as its key
        return {k: self._reduce(v) for k, v in losses.items()}


# inference


class ONNX:
    def __init__(
        self,
        *,
        onnx_path: str,
        output_names: List[str],
    ):
        self.ort_session = InferenceSession(onnx_path)
        self.output_names = output_names

    def inference(self, new_inputs: np_dict_type) -> np_dict_type:
        if self.ort_session is None:
            raise ValueError("`onnx_path` is not provided")
        ort_inputs = {
            node.name: to_standard(new_inputs[node.name])
            for node in self.ort_session.get_inputs()
        }
        return dict(zip(self.output_names, self.ort_session.run(None, ort_inputs)))


class InferenceProtocol:
    def __init__(
        self,
        *,
        onnx: Optional[ONNX] = None,
        model: Optional[ModelProtocol] = None,
        use_grad_in_predict: bool = False,
    ):
        self.onnx = onnx
        self.model = model
        if onnx is None and model is None:
            raise ValueError("either `onnx` or `model` should be provided")
        if onnx is not None and model is not None:
            raise ValueError("only one of `onnx` & `model` should be provided")
        self.use_grad_in_predict = use_grad_in_predict

    def get_outputs(
        self,
        loader: DataLoaderProtocol,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        metrics: Optional["MetricProtocol"] = None,
        loss: Optional[LossProtocol] = None,
        return_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        def _core() -> InferenceOutputs:
            results: Dict[str, Optional[List[np.ndarray]]] = {}
            metric_outputs_list: List[MetricsOutputs] = []
            loss_items: Dict[str, List[float]] = {}
            labels = []
            iterator = enumerate(loader)
            if use_tqdm:
                iterator = tqdm(list(iterator), desc="inference")
            requires_all_outputs = return_outputs
            if metrics is not None and metrics.requires_all:
                requires_all_outputs = True
            for i, batch in iterator:
                if i / len(loader) >= portion:
                    break
                np_batch = {
                    batch_key: None if batch_tensor is None else to_numpy(batch_tensor)
                    for batch_key, batch_tensor in batch.items()
                }
                if self.model is not None:
                    batch = to_device(batch, self.model.device)
                local_labels = batch[LABEL_KEY]
                if local_labels is not None:
                    if not isinstance(local_labels, np.ndarray):
                        local_labels = to_numpy(local_labels)
                    labels.append(local_labels)
                if self.onnx is not None:
                    local_outputs = self.onnx.inference(np_batch)
                else:
                    assert self.model is not None
                    with eval_context(self.model, use_grad=use_grad):
                        assert not self.model.training
                        local_outputs = self.model(
                            i,
                            batch,
                            state,
                            **shallow_copy_dict(kwargs),
                        )
                # gather outputs
                np_outputs: np_dict_type = {}
                for k, v in local_outputs.items():
                    if v is None:
                        continue
                    if isinstance(v, np.ndarray):
                        v_np = v
                    else:
                        v_np = to_numpy(v)
                    np_outputs[k] = v_np
                    if not requires_all_outputs:
                        results[k] = None
                    else:
                        results.setdefault(k, []).append(v_np)  # type: ignore
                # metrics
                if metrics is not None and not metrics.requires_all:
                    sub_outputs = metrics.evaluate(np_batch, np_outputs)
                    metric_outputs_list.append(sub_outputs)
                # loss
                if loss is not None:
                    with eval_context(loss, use_grad=use_grad):
                        local_losses = loss(local_outputs, batch)
                    for k, v in local_losses.items():
                        loss_items.setdefault(k, []).append(v.item())
            # gather outputs
            final_results: Dict[str, Union[np.ndarray, Any]]
            if not requires_all_outputs:
                final_results = {k: None for k in results}
            else:
                final_results = {
                    batch_key: np.vstack(batch_results)
                    for batch_key, batch_results in results.items()
                    if batch_results is not None
                }
            # gather metric outputs
            if metrics is None:
                metric_outputs = None
            elif metrics.requires_all:
                metric_outputs = metrics.evaluate(
                    {LABEL_KEY: np.vstack(labels)},
                    final_results,
                )
            else:
                scores = []
                metric_values: Dict[str, List[float]] = {}
                for sub_outputs in metric_outputs_list:
                    scores.append(sub_outputs.final_score)
                    for k, v in sub_outputs.metric_values.items():
                        metric_values.setdefault(k, []).append(v)
                metric_outputs = MetricsOutputs(
                    sum(scores) / len(scores),
                    {k: sum(vl) / len(vl) for k, vl in metric_values.items()},
                )
            return InferenceOutputs(
                final_results,
                None if not labels else np.vstack(labels),
                metric_outputs,
                None
                if not loss_items
                else {k: sum(v) / len(v) for k, v in loss_items.items()},
            )

        use_grad = kwargs.pop("use_grad", self.use_grad_in_predict)
        try:
            return _core()
        except:
            use_grad = self.use_grad_in_predict = True
            return _core()


# metrics


class MetricProtocol(ABC, WithRegister):
    d: Dict[str, Type["MetricProtocol"]] = metric_dict

    trainer: Any

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    @abstractmethod
    def is_positive(self) -> bool:
        pass

    @abstractmethod
    def _core(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        pass

    @property
    def requires_all(self) -> bool:
        return False

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(1, keepdims=True)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        metric = self._core(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        return MetricsOutputs(score, {self.__identifier__: metric})


__all__ = [
    "data_dict",
    "loader_dict",
    "loss_dict",
    "DataProtocol",
    "DataLoaderProtocol",
    "ModelProtocol",
    "TrainerState",
    "TrainerMonitor",
    "MonitorResults",
    "LossProtocol",
    "InferenceOutputs",
    "InferenceProtocol",
    "MetricsOutputs",
    "MetricProtocol",
]
