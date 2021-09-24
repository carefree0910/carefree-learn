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
from typing import Optional
from typing import NamedTuple
from onnxruntime import InferenceSession
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler

from .types import losses_type
from .types import np_dict_type
from .types import tensor_dict_type
from .constants import LOSS_KEY
from .constants import INPUT_KEY
from .constants import LABEL_KEY
from .misc.toolkit import to_numpy
from .misc.toolkit import to_device
from .misc.toolkit import to_standard
from .misc.toolkit import eval_context
from .misc.toolkit import get_world_size
from .misc.toolkit import WithRegister


dataset_dict: Dict[str, Type["DatasetProtocol"]] = {}
loader_dict: Dict[str, Type["DataLoaderProtocol"]] = {}
model_dict: Dict[str, Type["ModelProtocol"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["LossProtocol"]] = {}
metric_dict: Dict[str, Type["MetricProtocol"]] = {}


# data


class DatasetProtocol(ABC, WithRegister):
    d: Dict[str, Type["DatasetProtocol"]] = dataset_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataLoaderProtocol(ABC, WithRegister):
    d: Dict[str, Type["DataLoaderProtocol"]] = loader_dict
    data: DatasetProtocol
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "DataLoaderProtocol":
        pass

    @abstractmethod
    def __next__(self) -> tensor_dict_type:
        pass

    @abstractmethod
    def disable_shuffle(self) -> None:
        pass

    @abstractmethod
    def recover_shuffle(self) -> None:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def copy(self) -> "DataLoaderProtocol":
        return deepcopy(self)

    def temporarily_disable_shuffle(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, loader: DataLoaderProtocol):
                self.loader = loader

            def __enter__(self) -> None:
                self.loader.disable_shuffle()

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.loader.recover_shuffle()

        return _(self)


# model


class ModelProtocol(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["ModelProtocol"]] = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return list(self.parameters())[0].device

    def _init_with_trainer(self, trainer: Any) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.forward(0, batch)

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self.forward(batch_idx, batch)


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: Dict[str, float]


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
    custom_params_groups: bool = False
    custom_ddp_initialization: bool = False

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

    @staticmethod
    def params_groups(m: nn.Module) -> Any:
        pass

    def init_ddp(self, trainer: Any) -> None:
        pass

    def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        pass


# trainer


class TrainerState:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        *,
        num_epoch: int,
        max_epoch: int,
        fixed_steps: Optional[int] = None,
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
        self.batch_size = loader.batch_size * get_world_size()
        self.num_step_per_epoch = len(loader)
        self.num_epoch = num_epoch
        self.max_epoch = max_epoch
        self.fixed_steps = fixed_steps
        self.extension = extension
        self.enable_logging = enable_logging
        self.min_num_sample = min_num_sample
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_snapshot_per_epoch = num_snapshot_per_epoch
        self.num_step_per_log = num_step_per_log
        if num_step_per_snapshot is None:
            num_step_per_snapshot = max(1, int(len(loader) / num_snapshot_per_epoch))
            num_step_per_snapshot = min(max_step_per_snapshot, num_step_per_snapshot)
        self.num_step_per_snapshot = num_step_per_snapshot
        self.max_step_per_snapshot = max_step_per_snapshot

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_epoch": self.num_epoch,
            "max_epoch": self.max_epoch,
            "fixed_steps": self.fixed_steps,
            "extension": self.extension,
            "enable_logging": self.enable_logging,
            "min_num_sample": self.min_num_sample,
            "snapshot_start_step": self.snapshot_start_step,
            "max_snapshot_file": self.max_snapshot_file,
            "num_snapshot_per_epoch": self.num_snapshot_per_epoch,
            "num_step_per_log": self.num_step_per_log,
            "num_step_per_snapshot": self.num_step_per_snapshot,
            "max_step_per_snapshot": self.max_step_per_snapshot,
        }

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def should_train(self) -> bool:
        if self.fixed_steps is not None:
            return self.step < self.fixed_steps
        return self.epoch < self.num_epoch

    @property
    def should_terminate(self) -> bool:
        if self.fixed_steps is None:
            return False
        return self.step == self.fixed_steps

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

    def __init__(self, reduction: str = "mean", **kwargs: Any):
        super().__init__()
        self.config = kwargs
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
            labels: Dict[str, List[np.ndarray]] = {}
            iterator = enumerate(loader)
            if use_tqdm:
                iterator = tqdm(iterator, "inference", len(loader))
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
                requires_metrics = metrics is not None and not metrics.requires_all
                requires_np = requires_metrics or requires_all_outputs
                np_outputs: np_dict_type = {}
                for k, v in local_outputs.items():
                    if not requires_np:
                        results[k] = None
                        continue
                    if v is None:
                        continue
                    if isinstance(v, np.ndarray):
                        v_np = v
                    elif isinstance(v, torch.Tensor):
                        v_np = to_numpy(v)
                    elif isinstance(v, list):
                        if isinstance(v[0], np.ndarray):
                            v_np = v
                        else:
                            v_np = list(map(to_numpy, v))
                    else:
                        raise ValueError(f"unrecognized value ({k}={type(v)}) occurred")
                    np_outputs[k] = v_np
                    if not requires_all_outputs:
                        results[k] = None
                    else:
                        results.setdefault(k, []).append(v_np)  # type: ignore
                if requires_np:
                    for k, v in batch.items():
                        if k == INPUT_KEY or v is None:
                            continue
                        if k != LABEL_KEY and len(v.shape) > 2:
                            continue
                        if not isinstance(v, np.ndarray):
                            v = to_numpy(v)
                        labels.setdefault(k, []).append(v)
                # metrics
                if requires_metrics:
                    sub_outputs = metrics.evaluate(np_batch, np_outputs)  # type: ignore
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
                    if isinstance(batch_results[0], np.ndarray)
                    else [
                        np.vstack([batch[i] for batch in batch_results])
                        for i in range(len(batch_results[0]))
                    ]
                    for batch_key, batch_results in results.items()
                    if batch_results is not None
                }
            # gather metric outputs
            if metrics is None:
                metric_outputs = None
            elif metrics.requires_all:
                metric_outputs = metrics.evaluate(
                    {k: np.vstack(v) for k, v in labels.items()},
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
        with loader.temporarily_disable_shuffle():
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
    "dataset_dict",
    "loader_dict",
    "loss_dict",
    "DatasetProtocol",
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
