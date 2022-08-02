import os
import math
import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from copy import deepcopy
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import NamedTuple
from cftool.misc import filter_kw
from cftool.misc import lock_manager
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cftool.misc import get_num_positional_args
from cftool.misc import WithRegister
from cftool.array import to_numpy
from cftool.array import to_device
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .types import losses_type
from .types import configs_type
from .constants import LOSS_KEY
from .constants import INPUT_KEY
from .constants import LABEL_KEY
from .constants import INFO_PREFIX
from .constants import WARNING_PREFIX
from .constants import PREDICTIONS_KEY
from .constants import BATCH_INDICES_KEY
from .constants import ORIGINAL_LABEL_KEY
from .misc.toolkit import eval_context
from .misc.toolkit import get_world_size
from .misc.toolkit import fix_denormal_states
from .misc.toolkit import ONNX

try:
    import onnx
except:
    onnx = None
try:
    from onnxsim import simplify as onnx_simplify

    def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
        initializer_names = [x.name for x in model.graph.initializer]
        return [inp for inp in model.graph.input if inp.name not in initializer_names]

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = [inp.name for inp in get_inputs(model)]
        return input_names

except:
    onnx_simplify = get_input_names = None  # type: ignore


dataset_dict: Dict[str, Type["DatasetProtocol"]] = {}
loader_dict: Dict[str, Type["DataLoaderProtocol"]] = {}
model_dict: Dict[str, Type["ModelProtocol"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["LossProtocol"]] = {}
metric_dict: Dict[str, Type["MetricProtocol"]] = {}


# data


class DatasetProtocol(WithRegister["DatasetProtocol"], metaclass=ABCMeta):
    d = dataset_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class DataLoaderProtocol(WithRegister["DataLoaderProtocol"], metaclass=ABCMeta):
    d = loader_dict
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


def _forward(
    m: nn.Module,
    batch_idx: int,
    batch: tensor_dict_type,
    general_input_key: str,
    state: Optional["TrainerState"] = None,
    *,
    general_output_key: str = PREDICTIONS_KEY,
    **kwargs: Any,
) -> tensor_dict_type:
    fn = m.forward
    if check_requires(fn, "general_output_key"):
        kwargs["general_output_key"] = general_output_key
    kw = filter_kw(fn, kwargs)
    args: List[Any] = []
    if check_requires(fn, "batch_idx"):
        args.append(batch_idx)
    if get_num_positional_args(fn) > 0:
        args.append(batch if check_requires(fn, "batch") else batch[general_input_key])
    if check_requires(fn, "state"):
        args.append(state)
    rs = m(*args, **kw)
    if not isinstance(rs, dict):
        rs = {general_output_key: rs}
    return rs


class WithDeviceMixin:
    parameters: Callable[["WithDeviceMixin"], Iterator[nn.Parameter]]

    @property
    def device(self) -> torch.device:
        params = list(self.parameters())
        return torch.device("cpu") if not params else params[0].device


class ModelProtocol(
    WithDeviceMixin,
    nn.Module,
    WithRegister["ModelProtocol"],
    metaclass=ABCMeta,
):
    d = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

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

    def to_onnx(
        self,
        export_folder: str,
        input_sample: tensor_dict_type,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        onnx_file: str = "model.onnx",
        opset: int = 11,
        simplify: bool = True,
        forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "ModelProtocol":
        # prepare
        device = self.device
        model = self.cpu()
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        onnx_forward = forward_fn or model.onnx_forward
        input_names = sorted(input_sample.keys())
        if output_names is None:
            if forward_fn is not None:
                msg = "`output_names` should be provided when `forward_fn` is provided"
                raise ValueError(msg)
            with eval_context(model):
                forward_results = onnx_forward(shallow_copy_dict(input_sample))
            if not isinstance(forward_results, dict):
                forward_results = {PREDICTIONS_KEY: forward_results}
            output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        # export
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                rs = onnx_forward(batch)
                if isinstance(rs, torch.Tensor):
                    return {k: rs for k in output_names}  # type: ignore
                return {k: rs[k] for k in output_names}  # type: ignore

        with lock_manager(base_folder, []) as lock:
            onnx_path = os.path.join(export_folder, onnx_file)
            lock._stuffs = [onnx_path]
            os.makedirs(export_folder, exist_ok=True)
            m_onnx = ONNXWrapper()
            original_states = model.state_dict()
            fixed_states = fix_denormal_states(original_states, verbose=verbose)
            with eval_context(m_onnx):
                model.load_state_dict(fixed_states)
                torch.onnx.export(
                    m_onnx,
                    ({k: input_sample[k] for k in input_names}, {}),
                    onnx_path,
                    **shallow_copy_dict(kwargs),
                )
                model.load_state_dict(original_states)
                if not simplify:
                    return self.to(device)
                if onnx is None:
                    print(
                        f"{WARNING_PREFIX}`onnx` is not installed, "
                        f"so the exported onnx model will not be simplified"
                    )
                    return self.to(device)
                if onnx_simplify is None or get_input_names is None:
                    print(
                        f"{WARNING_PREFIX}`onnx-simplifier` is not installed, "
                        f"so the exported onnx model will not be simplified"
                    )
                    return self.to(device)
                try:
                    onnx_model = onnx.load(onnx_path)
                    final_input_names = get_input_names(onnx_model)
                    np_sample = {
                        name: to_numpy(tensor)
                        for name, tensor in input_sample.items()
                        if name in final_input_names
                    }
                    model_simplified, check = onnx_simplify(
                        onnx_model,
                        input_data=np_sample,
                        dynamic_input_shape=bool(dynamic_axes),
                    )
                except Exception as err:
                    if verbose:
                        print(f"{WARNING_PREFIX}Failed to simplify ONNX model ({err})")
                    model_simplified = None
                    check = False
                if verbose:
                    tag = " " if check else " not "
                    print(f"{INFO_PREFIX}Simplified ONNX model is{tag}validated!")
                if check and model_simplified is not None:
                    onnx.save(model_simplified, onnx_path)
        return self.to(device)


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

    # TODO : Add `forward_kwargs`
    @abstractmethod
    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        pass

    def params_groups(self, m: nn.Module) -> Any:
        pass

    def init_ddp(self) -> None:
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
        max_step_per_snapshot: int = 1000,
        min_snapshot_epoch_gap: int = 0,
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
        self.min_snapshot_epoch_gap = min_snapshot_epoch_gap
        self._previous_snapshot_epoch = 0

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    def update_snapshot_epoch(self) -> None:
        self._previous_snapshot_epoch = self.epoch

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
    def can_snapshot(self) -> bool:
        if self.is_terminate:
            return True
        return self.epoch - self._previous_snapshot_epoch >= self.min_snapshot_epoch_gap

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


class TrainerMonitor(WithRegister["TrainerMonitor"], metaclass=ABCMeta):
    d = monitor_dict

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


LossType = TypeVar("LossType", bound="LossProtocol", covariant=True)


class LossProtocol(nn.Module, WithRegister[LossType], metaclass=ABCMeta):
    d = loss_dict
    placeholder_key = "[PLACEHOLDER]"

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

    @classmethod
    def parse(cls, name: str) -> str:
        if ":" not in name:
            return name
        split = name.split(":")
        if split[-2] != AuxLoss.identifier:
            for prefix, base in multi_prefix_mapping.items():
                if name.startswith(f"{prefix}:"):
                    loss_names = name.split(":")[1].split(",")
                    return base.register_(loss_names)
            return name
        loss_name = cls.parse(":".join(split[:-2]))
        aux_names = split[-1].split(",")
        return AuxLoss.register_(loss_name, aux_names)


multi_prefix_mapping: Dict[str, Type["MultiLoss"]] = {}


class MultiLoss(LossProtocol, metaclass=ABCMeta):
    prefix: str

    names: Union[str, List[str]]
    base_losses: nn.ModuleList

    def _init_config(self) -> None:
        base_losses: List[LossProtocol]
        if isinstance(self.names, str):
            base_losses = [LossProtocol.make(self.names, self.config)]
        else:
            base_losses = [
                LossProtocol.make(name, self.config.get(name, {}))
                for name in self.names
            ]
        self.base_losses = nn.ModuleList(base_losses)

    @staticmethod
    def _inject(key: str, base_losses: losses_type, all_losses: losses_type) -> None:
        if isinstance(base_losses, dict):
            base_losses = base_losses[LOSS_KEY]
        all_losses[key] = base_losses

    @classmethod
    def register_(
        cls,
        base_loss_names: Union[str, List[str]],
        *,
        tag: Optional[str] = None,
    ) -> str:
        if tag is None:
            if isinstance(base_loss_names, str):
                tag = f"{cls.prefix}_{base_loss_names}"
            else:
                tag = f"{cls.prefix}_{'_'.join(base_loss_names)}"
        if tag in cls.d:
            return tag

        @cls.register(tag)  # type: ignore
        class _(cls):  # type: ignore
            names = base_loss_names

        return tag

    @classmethod
    def record_prefix(cls) -> Callable[[Type["MultiLoss"]], Type["MultiLoss"]]:
        def _(cls_: Type[MultiLoss]) -> Type[MultiLoss]:
            global multi_prefix_mapping
            multi_prefix_mapping[cls_.prefix] = cls_
            return cls_

        return _


class AuxLoss(LossProtocol):
    identifier: str = "aux"
    main_loss_key: str = "main"

    loss: LossProtocol
    loss_name: str
    aux_names: List[str]

    def _init_config(self) -> None:
        self.loss = LossProtocol.make(self.loss_name, self.config)

    @staticmethod
    def _convert(losses: losses_type) -> torch.Tensor:
        if isinstance(losses, dict):
            return losses[LOSS_KEY]
        return losses

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        main_losses = self.loss._core(forward_results, batch, state, **kwargs)
        losses = {self.main_loss_key: self._convert(main_losses)}
        for name in self.aux_names:
            losses[name] = self._convert(
                self.loss._core(
                    {PREDICTIONS_KEY: forward_results[name]},
                    {LABEL_KEY: batch[name]},
                    state,
                    **kwargs,
                )
            )
        losses[LOSS_KEY] = sum(losses.values())
        return losses

    @classmethod
    def register_(
        cls,
        base_loss_name: str,
        aux_loss_names: List[str],
        *,
        tag: Optional[str] = None,
    ) -> str:
        for name in aux_loss_names:
            if name == cls.main_loss_key:
                raise ValueError(f"should not use '{cls.main_loss_key}' as aux name")
        if tag is None:
            tag = f"{base_loss_name}_{cls.identifier}_{'_'.join(aux_loss_names)}"
        if tag in cls.d:
            return tag

        @cls.register(tag)  # type: ignore
        class _(cls):  # type: ignore
            loss_name = base_loss_name
            aux_names = aux_loss_names

        return tag


@LossProtocol.register(LossProtocol.placeholder_key)
class PlaceholderLoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        raise ValueError("`forward` should not be called in `PlaceholderLoss`")


# inference


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
        stack_outputs: bool = True,
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
                    local_outputs = self.onnx.predict(np_batch)
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
                        if (
                            k == INPUT_KEY
                            or k == ORIGINAL_LABEL_KEY
                            or k.endswith(BATCH_INDICES_KEY)
                        ):
                            continue
                        if v is None:
                            continue
                        if k != LABEL_KEY and len(v.shape) > 2:
                            continue
                        if not isinstance(v, np.ndarray):
                            v = to_numpy(v)
                        labels.setdefault(k, []).append(v)
                # metrics
                if requires_metrics:
                    sub_outputs = metrics.evaluate(np_batch, np_outputs, loader)  # type: ignore
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
                    batch_key: batch_results
                    if not stack_outputs
                    else np.vstack(batch_results)
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
                    loader,
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

            target_labels = labels.get(LABEL_KEY, [])
            return InferenceOutputs(
                final_results,
                None if not target_labels else np.vstack(target_labels),
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


class MetricProtocol(WithRegister["MetricProtocol"], metaclass=ABCMeta):
    d = metric_dict

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

    @classmethod
    def fuse(
        cls,
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> "MetricProtocol":
        metrics = MetricProtocol.make_multiple(names, configs)
        if isinstance(metrics, MetricProtocol):
            return metrics
        return MultipleMetrics(metrics, weights=metric_weights)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        metric = self._core(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        return MetricsOutputs(score, {self.__identifier__: metric})


class MultipleMetrics(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    @property
    def requires_all(self) -> bool:
        return any(metric.requires_all for metric in self.metrics)

    def _core(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[MetricProtocol],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}
        self.__identifier__ = " | ".join(m.__identifier__ for m in metric_list)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(np_batch, np_outputs, loader)
            w = self.weights.get(metric.__identifier__, 1.0)
            weights.append(w)
            scores.append(metric_outputs.final_score * w)
            metrics_values.update(metric_outputs.metric_values)
        return MetricsOutputs(sum(scores) / sum(weights), metrics_values)


__all__ = [
    "_forward",
    "dataset_dict",
    "loader_dict",
    "loss_dict",
    "DatasetProtocol",
    "DataLoaderProtocol",
    "ModelProtocol",
    "ModelWithCustomSteps",
    "StepOutputs",
    "TrainerState",
    "TrainerMonitor",
    "MonitorResults",
    "LossProtocol",
    "MultiLoss",
    "AuxLoss",
    "ONNX",
    "InferenceOutputs",
    "InferenceProtocol",
    "MetricsOutputs",
    "MetricProtocol",
    "MultipleMetrics",
]
