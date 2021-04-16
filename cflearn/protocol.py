import math
import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler

from .constants import *
from .types import losses_type
from .types import np_dict_type
from .types import tensor_dict_type
from .misc.toolkit import to_numpy
from .misc.toolkit import to_device
from .misc.toolkit import eval_context


data_dict: Dict[str, Type["DataProtocol"]] = {}
loader_dict: Dict[str, Type["DataLoaderProtocol"]] = {}
model_dict: Dict[str, Type["ModelProtocol"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["LossProtocol"]] = {}
metric_dict: Dict[str, Type["MetricProtocol"]] = {}


class WithRegister:
    d: Dict[str, Type]
    __identifier__: str

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


# data


class DataProtocol(ABC, WithRegister):
    d: Dict[str, Type] = data_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataLoaderProtocol(ABC, WithRegister):
    d: Dict[str, Type] = loader_dict
    data: DataProtocol
    batch_size: int

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __iter__(self) -> "DataLoaderProtocol":
        pass

    @abstractmethod
    def __next__(self) -> tensor_dict_type:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)


# model


class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type] = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: "TrainerState",
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass


# trainer


class TrainerState:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        *,
        num_epoch: int,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 5,
        num_snapshot_per_epoch: int = 2,
        num_step_per_log: int = 350,
        num_step_per_snapshot: Optional[int] = None,
    ):
        self.step = self.epoch = 0
        self.batch_size = loader.batch_size
        self.num_step_per_epoch = len(loader)
        self.num_epoch = num_epoch
        self.enable_logging = enable_logging
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_step_per_log = num_step_per_log
        if num_step_per_snapshot is None:
            num_step_per_snapshot = int(len(loader) / num_snapshot_per_epoch)
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
        min_period = math.ceil(self.num_step_per_log / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step

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
    d: Dict[str, Type] = monitor_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def snapshot(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def check_terminate(self, new_score: float) -> bool:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global monitor_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, monitor_dict, before_register=before)


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    outputs: Optional["InferenceOutputs"]
    metric_outputs: Optional["MetricsOutputs"]


# loss


class LossProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type] = loss_dict

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
        **kwargs: Any,
    ) -> losses_type:
        # return losses without reduction
        pass

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
    ) -> tensor_dict_type:
        losses = self._core(forward_results, batch)
        if isinstance(losses, torch.Tensor):
            return {LOSS_KEY: self._reduce(losses)}
        # requires returns a value with LOSS_KEY as its key
        return {k: self._reduce(v) for k, v in losses.items()}

    @classmethod
    def make(
        cls,
        name: str,
        config: Dict[str, Any],
        reduction: str = "mean",
    ) -> "LossProtocol":
        return loss_dict[name](config, reduction)


# inference


class InferenceOutputs(NamedTuple):
    forward_results: np_dict_type
    labels: Optional[np.ndarray]
    loss_items: Optional[Dict[str, float]]


class InferenceProtocol:
    def __init__(
        self,
        model: ModelProtocol,
        *,
        use_grad_in_predict: bool = False,
    ):
        self.model = model
        self.use_grad_in_predict = use_grad_in_predict

    def get_outputs(
        self,
        device: torch.device,
        loader: DataLoaderProtocol,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        loss: Optional[LossProtocol] = None,
        return_outputs: bool = True,
        **kwargs: Any,
    ) -> InferenceOutputs:
        def _core() -> InferenceOutputs:
            results: Dict[str, Optional[List[np.ndarray]]] = {}
            loss_items: Dict[str, List[float]] = {}
            labels = []
            for i, batch in enumerate(loader):
                if i / len(loader) >= portion:
                    break
                batch = to_device(batch, device)
                local_labels = batch[LABEL_KEY]
                if local_labels is not None:
                    if not isinstance(local_labels, np.ndarray):
                        local_labels = to_numpy(local_labels)
                    labels.append(local_labels)
                assert self.model is not None
                with eval_context(self.model, use_grad=use_grad):
                    assert not self.model.training
                    local_results = self.model(
                        i,
                        batch,
                        state,
                        **shallow_copy_dict(kwargs),
                    )
                if loss is None:
                    local_losses = None
                else:
                    local_losses = loss(local_results, batch)
                for k, v in local_results.items():
                    if v is None:
                        continue
                    v_np = to_numpy(v)
                    if not return_outputs:
                        results[k] = None
                    else:
                        results.setdefault(k, []).append(v_np)  # type: ignore
                if local_losses is not None:
                    for k, v in local_losses.items():
                        loss_items.setdefault(k, []).append(v.item())

            final_results: Dict[str, Union[np.ndarray, Any]]
            if not return_outputs:
                final_results = {k: None for k in results}
            else:
                final_results = {
                    k: np.vstack(v) for k, v in results.items() if v is not None
                }

            return InferenceOutputs(
                final_results,
                None if not labels else np.vstack(labels),
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


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]


class MetricProtocol(ABC, WithRegister):
    d: Dict[str, Type] = metric_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(1, keepdims=True)

    @property
    @abstractmethod
    def is_positive(self) -> bool:
        pass

    @abstractmethod
    def _core(self, outputs: InferenceOutputs) -> float:
        pass

    def evaluate(self, outputs: InferenceOutputs) -> MetricsOutputs:
        metric = self._core(outputs)
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
