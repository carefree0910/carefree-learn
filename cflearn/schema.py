import os
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
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import NamedTuple
from dataclasses import dataclass
from torch.optim import Optimizer
from cftool.misc import filter_kw
from cftool.misc import print_info
from cftool.misc import print_warning
from cftool.misc import safe_execute
from cftool.misc import lock_manager
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cftool.misc import get_num_positional_args
from cftool.misc import DataClassBase
from cftool.misc import WithRegister
from cftool.array import to_numpy
from cftool.array import to_device
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from torch.optim.lr_scheduler import _LRScheduler

from .types import losses_type
from .types import configs_type
from .types import sample_weights_type
from .constants import LOSS_KEY
from .constants import INPUT_KEY
from .constants import LABEL_KEY
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


dataset_dict: Dict[str, Type["IDataset"]] = {}
loader_dict: Dict[str, Type["IDataLoader"]] = {}
model_dict: Dict[str, Type["IDLModel"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["ILoss"]] = {}
multi_prefix_mapping: Dict[str, Type["MultiLoss"]] = {}
metric_dict: Dict[str, Type["_IMetric"]] = {}
callback_dict: Dict[str, Type["TrainerCallback"]] = {}

TLoss = TypeVar("TLoss", bound="ILoss", covariant=True)
TDataModule = TypeVar("TDataModule")


# data


class IDataset(WithRegister["IDataset"], metaclass=ABCMeta):
    d = dataset_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class IDataLoader(WithRegister["IDataLoader"], metaclass=ABCMeta):
    d = loader_dict
    data: IDataset
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "IDataLoader":
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

    def copy(self) -> "IDataLoader":
        return deepcopy(self)

    def temporarily_disable_shuffle(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, loader: IDataLoader):
                self.loader = loader

            def __enter__(self) -> None:
                self.loader.disable_shuffle()

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.loader.recover_shuffle()

        return _(self)


class IDataModule(ABC):
    id_file: str
    info_name: str
    data_folder: str
    package_folder: str

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare(self: TDataModule, sample_weights: sample_weights_type) -> TDataModule:
        pass

    @abstractmethod
    def initialize(self) -> Any:
        pass

    @abstractmethod
    def save_info(self, folder: str) -> None:
        pass

    @abstractmethod
    def save(self, folder: str) -> None:
        pass


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
        params = list(self.parameters())  # type: ignore
        return torch.device("cpu") if not params else params[0].device


class IDLModel(
    WithDeviceMixin,
    nn.Module,
    WithRegister["IDLModel"],
    metaclass=ABCMeta,
):
    d = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def _init_with_trainer(self, trainer: "ITrainer") -> None:
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
    ) -> "IDLModel":
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
                    print_warning(
                        "`onnx` is not installed, "
                        "so the exported onnx model will not be simplified"
                    )
                    return self.to(device)
                if onnx_simplify is None or get_input_names is None:
                    print_warning(
                        "`onnx-simplifier` is not installed, "
                        "so the exported onnx model will not be simplified"
                    )
                    return self.to(device)
                try:
                    onnx_model = onnx.load(onnx_path)
                    final_input_names = get_input_names(onnx_model)
                    model_simplified, check = onnx_simplify(
                        onnx_model,
                        test_input_shapes={
                            name: tensor.shape
                            for name, tensor in input_sample.items()
                            if name in final_input_names
                        },
                    )
                except Exception as err:
                    if verbose:
                        print_warning(f"Failed to simplify ONNX model ({err})")
                    model_simplified = None
                    check = False
                if verbose:
                    tag = " " if check else " not "
                    print_info(f"Simplified ONNX model is{tag}validated!")
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


class ModelWithCustomSteps(IDLModel, metaclass=ABCMeta):
    custom_train_step: bool = True
    custom_evaluate_step: bool = True
    custom_params_groups: bool = False
    custom_ddp_initialization: bool = False

    @abstractmethod
    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: "ITrainer",
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        pass

    # TODO : Add `forward_kwargs`
    @abstractmethod
    def evaluate_step(
        self,
        loader: IDataLoader,
        portion: float,
        trainer: "ITrainer",
    ) -> MetricsOutputs:
        pass

    def params_groups(self, m: nn.Module) -> Any:
        pass

    def init_ddp(self) -> None:
        pass

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        pass


# trainer types


class TrainerState:
    def __init__(
        self,
        loader: IDataLoader,
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


class ILoss(nn.Module, WithRegister[TLoss], metaclass=ABCMeta):
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


class MultiLoss(ILoss, metaclass=ABCMeta):
    prefix: str

    names: Union[str, List[str]]
    base_losses: nn.ModuleList

    def _init_config(self) -> None:
        base_losses: List[ILoss]
        if isinstance(self.names, str):
            base_losses = [ILoss.make(self.names, self.config)]
        else:
            base_losses = [
                ILoss.make(name, self.config.get(name, {})) for name in self.names
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


class AuxLoss(ILoss):
    identifier: str = "aux"
    main_loss_key: str = "main"

    loss: ILoss
    loss_name: str
    aux_names: List[str]

    def _init_config(self) -> None:
        self.loss = ILoss.make(self.loss_name, self.config)

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


@ILoss.register(ILoss.placeholder_key)
class PlaceholderLoss(ILoss):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        raise ValueError("`forward` should not be called in `PlaceholderLoss`")


# inference


class IInference:
    def __init__(
        self,
        *,
        onnx: Optional[ONNX] = None,
        model: Optional[IDLModel] = None,
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
        loader: IDataLoader,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        metrics: Optional["_IMetric"] = None,
        loss: Optional[ILoss] = None,
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
                        kw = filter_kw(self.model.forward, shallow_copy_dict(kwargs))
                        local_outputs = self.model(i, batch, state, **kw)
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


class _IMetric(WithRegister["_IMetric"], metaclass=ABCMeta):
    d = metric_dict

    trainer: "ITrainer"

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
        loader: Optional[IDataLoader],
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
    ) -> "_IMetric":
        metrics = _IMetric.make_multiple(names, configs)
        if isinstance(metrics, _IMetric):
            return metrics
        return _MultipleMetrics(metrics, weights=metric_weights)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        metric = self._core(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        return MetricsOutputs(score, {self.__identifier__: metric})


class _MultipleMetrics(_IMetric):
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
        loader: Optional[IDataLoader],
    ) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[_IMetric],
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
        loader: Optional[IDataLoader] = None,
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


# trainer interface


class DeviceInfo(NamedTuple):
    cuda: Optional[str]
    rank: Optional[int]

    @property
    def device(self) -> torch.device:
        if self.rank is not None:
            return torch.device(f"cuda:{self.rank}")
        return torch.device("cpu" if self.cuda is None else f"cuda:{self.cuda}")


class OptimizerPack(NamedTuple):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None


class TqdmSettings(NamedTuple):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    in_distributed: bool = False
    position: int = 0
    desc: str = "epoch"


class TrainerCallback(WithRegister["TrainerCallback"]):
    d = callback_dict
    is_rank_0: bool = True

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def initialize(self) -> None:
        pass

    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def mutate_train_loss_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def before_loop(self, trainer: "ITrainer") -> None:
        pass

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        pass

    def log_metrics_msg(
        self,
        metric_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
    ) -> None:
        pass

    def log_artifacts(self, trainer: "ITrainer") -> None:
        pass

    def after_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        pass

    def after_monitor(
        self,
        monitor_results: MonitorResults,
        state: TrainerState,
    ) -> None:
        pass

    def finalize(self, trainer: "ITrainer") -> None:
        pass


class ITrainer(ABC):
    loss: ILoss
    model: IDLModel
    metrics: Optional[_IMetric]
    monitors: List[TrainerMonitor]
    callbacks: List[TrainerCallback]
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]

    state: TrainerState
    device_info: DeviceInfo
    train_loader: IDataLoader
    train_loader_copy: IDataLoader
    valid_loader: Optional[IDataLoader]
    inference: IInference

    workplace: str
    checkpoint_folder: Optional[str]

    tqdm_settings: TqdmSettings
    state_config: Dict[str, Any]
    num_epoch: int
    max_epoch: int
    fixed_steps: Optional[int]
    valid_portion: float
    use_amp: bool
    use_zero: bool
    clip_norm: float
    grad_scaler: torch.cuda.amp.GradScaler

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def use_tqdm_in_validation(self) -> bool:
        pass

    @property
    @abstractmethod
    def validation_loader(self) -> IDataLoader:
        pass

    @property
    @abstractmethod
    def input_sample(self) -> tensor_dict_type:
        pass

    @property
    @abstractmethod
    def has_checkpoint_folder(self) -> bool:
        pass

    @property
    @abstractmethod
    def model_has_custom_steps(self) -> bool:
        pass

    @property
    @abstractmethod
    def model_for_training(self) -> nn.Module:
        pass

    # init

    @abstractmethod
    def _init_ddp(self) -> None:
        pass

    @abstractmethod
    def _init_finetune(self) -> None:
        pass

    @abstractmethod
    def default_lr_configs(
        self,
        optimizer: Optimizer,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def _define_optimizer(self, pack: OptimizerPack) -> Optimizer:
        pass

    @abstractmethod
    def _define_scheduler(self, optimizer: Optimizer, pack: OptimizerPack) -> None:
        pass

    @abstractmethod
    def _init_optimizers(self) -> None:
        pass

    # core

    @abstractmethod
    def post_loss_step(self, loss_dict: tensor_dict_type) -> None:
        pass

    @abstractmethod
    def weighted_loss_score(self, loss_items: Dict[str, float]) -> float:
        pass

    @abstractmethod
    def clip_norm_step(self) -> None:
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        pass

    @abstractmethod
    def scheduler_step(self) -> None:
        pass

    @abstractmethod
    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        pass

    @abstractmethod
    def _monitor_step(self, step_outputs: StepOutputs) -> MonitorResults:
        pass

    @abstractmethod
    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        pass

    @abstractmethod
    def fit(
        self,
        data: IDataModule,
        loss: ILoss,
        model: IDLModel,
        inference: IInference,
        *,
        config_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        cuda: Optional[str] = None,
    ) -> "ITrainer":
        pass

    @abstractmethod
    def get_metrics(
        self,
        *,
        portion: float = 1.0,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        pass


# configs


@dataclass
class TrainerConfig(DataClassBase):
    state_config: Optional[Dict[str, Any]] = None
    num_epoch: int = 40
    max_epoch: int = 1000
    fixed_epoch: Optional[int] = None
    fixed_steps: Optional[int] = None
    log_steps: Optional[int] = None
    valid_portion: float = 1.0
    amp: bool = False
    clip_norm: float = 0.0
    metric_names: Optional[Union[str, List[str]]] = None
    metric_configs: configs_type = None
    metric_weights: Optional[Dict[str, float]] = None
    use_losses_as_metrics: Optional[bool] = None
    loss_metrics_weights: Optional[Dict[str, float]] = None
    recompute_train_losses_in_eval: bool = True
    monitor_names: Optional[Union[str, List[str]]] = None
    monitor_configs: Optional[Dict[str, Any]] = None
    auto_callback: bool = True
    callback_names: Optional[Union[str, List[str]]] = None
    callback_configs: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    update_scheduler_per_epoch: bool = False
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None
    use_zero: bool = False
    workplace: str = "_logs"
    data_info_name: str = "data_info"
    metrics_log_file: str = "metrics.txt"
    finetune_config: Optional[Dict[str, Any]] = None
    tqdm_settings: Optional[Dict[str, Any]] = None


@dataclass
class Config(TrainerConfig):
    loss_name: Optional[str] = None
    loss_config: Optional[Dict[str, Any]] = None
    in_loading: bool = False
    allow_no_loss: bool = False
    cudnn_benchmark: bool = False

    def to_debug(self) -> None:
        self.fixed_steps = 1
        self.valid_portion = 1.0e-4

    @property
    def trainer_config(self) -> TrainerConfig:
        return safe_execute(TrainerConfig, self.asdict())


@dataclass
class _DLConfig:
    model_name: str
    model_config: Optional[Dict[str, Any]] = None


@dataclass
class DLConfig(Config, _DLConfig):
    pass


__all__ = [
    "_forward",
    "dataset_dict",
    "loader_dict",
    "loss_dict",
    "multi_prefix_mapping",
    "metric_dict",
    "callback_dict",
    "IDataset",
    "IDataLoader",
    "IDLModel",
    "ModelWithCustomSteps",
    "StepOutputs",
    "TrainerState",
    "TrainerMonitor",
    "MonitorResults",
    "ILoss",
    "MultiLoss",
    "AuxLoss",
    "ONNX",
    "InferenceOutputs",
    "IInference",
    "MetricsOutputs",
    "_IMetric",
    "_MultipleMetrics",
    "DeviceInfo",
    "OptimizerPack",
    "TqdmSettings",
    "TrainerCallback",
    "ITrainer",
    "TrainerConfig",
    "Config",
    "DLConfig",
]
