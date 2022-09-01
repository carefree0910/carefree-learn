import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import safe_execute
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import get_num_positional_args
from cftool.array import to_device
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim.optimizer import Optimizer

from ..toolkit import toggle_optimizer
from ..toolkit import Initializer
from ...types import losses_type
from ...protocol import _forward
from ...protocol import ITrainer
from ...protocol import StepOutputs
from ...protocol import ILoss
from ...protocol import TrainerState
from ...protocol import IDLModel
from ...protocol import _IMetric
from ...protocol import MetricsOutputs
from ...protocol import WithDeviceMixin
from ...protocol import IDataLoader
from ...protocol import ModelWithCustomSteps
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ...data.cv import Transforms


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


class IMetric:
    @property
    @abstractmethod
    def is_positive(self) -> bool:
        pass

    @abstractmethod
    def forward(self, *args: Any) -> float:
        """
        Possible argument patterns:
        * predictions, labels
        * predictions, labels, loader
            * The above two patterns could have different namings, e.g.
                * outputs, targets
                * y_preds, y_true, loader
                * ...
            * Notice that they should not be `np_outputs` & `np_batch`
        * np_outputs, np_batch
        * np_outputs, np_batch, loader
            * The above two patterns should have the exact namings
        """

    @property
    def requires_all(self) -> bool:
        return False


class ITransform:
    @abstractmethod
    def forward(self, sample: Dict[str, Any]) -> tensor_dict_type:
        pass


metric_type = Type[IMetric]
transform_type = Type[ITransform]


def register_metric(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable[[metric_type], metric_type]:
    def _core(metric_base: metric_type) -> metric_type:
        @_IMetric.register(name, allow_duplicate=allow_duplicate)
        class _(_IMetric):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.core = metric_base(*args, **kwargs)

            @property
            def is_positive(self) -> bool:
                return self.core.is_positive

            @property
            def requires_all(self) -> bool:
                return self.core.requires_all

            def _core(
                self,
                np_batch: np_dict_type,
                np_outputs: np_dict_type,
                loader: Optional[IDataLoader],
            ) -> float:
                args: List[Any] = []
                fn = self.core.forward
                if check_requires(fn, "np_outputs"):
                    args.append(np_outputs)
                else:
                    args.append(np_outputs[PREDICTIONS_KEY])
                if check_requires(fn, "np_batch"):
                    args.append(np_batch)
                else:
                    args.append(np_batch[LABEL_KEY])
                if check_requires(fn, "loader"):
                    args.append(loader)
                return self.core.forward(*args)

        return metric_base

    return _core


def register_module(
    name: str,
    *,
    allow_duplicate: bool = False,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _core(m: Type[nn.Module]) -> Type[nn.Module]:
        @IDLModel.register(name, allow_duplicate=allow_duplicate)
        class _(*bases):  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.core = m(*args, **kwargs)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                return _forward(self.core, batch_idx, batch, INPUT_KEY, state, **kwargs)

        return m

    bases = (pre_bases or []) + [IDLModel] + (post_bases or [])
    return _core


class CustomTrainStepLoss(NamedTuple):
    loss: Tensor
    losses: Dict[str, float]


class CustomTrainStep(ABC):
    def __init__(
        self,
        scope: str = "all",
        *,
        num_forward: int = 1,
        enable_toggle_optimizer: bool = True,
    ) -> None:
        self.scope = scope
        self.num_forward = num_forward
        self.enable_toggle_optimizer = enable_toggle_optimizer

    @property
    def requires_new_forward(self) -> bool:
        return False

    @property
    def requires_scheduler_step(self) -> bool:
        return False

    def should_skip(self, m: "CustomModule", state: TrainerState) -> bool:
        return False

    @abstractmethod
    def loss_fn(
        self,
        m: "CustomModule",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        pass


class CustomModule(WithDeviceMixin, nn.Module):
    @property
    def train_steps(self) -> Optional[List[CustomTrainStep]]:
        return None

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return _forward(self, 0, batch, INPUT_KEY)

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        _forward(self, batch_idx, batch, INPUT_KEY)

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        optimizers: Dict[str, Optimizer],
        use_amp: bool,
        grad_scaler: GradScaler,
        clip_norm_fn: Callable[[], None],
        update_fn: Callable[[Tensor, Optimizer], None],
        scheduler_step_fn: Callable[[], None],
        trainer: ITrainer,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        pass

    def evaluate_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: ITrainer,
    ) -> MetricsOutputs:
        pass

    def params_groups(self, m: nn.Module) -> Any:
        pass

    def init_ddp(self) -> None:
        pass

    def init_with_trainer(self, trainer: ITrainer) -> None:
        pass

    def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        pass


def get_update_fn(trainer: ITrainer) -> Callable[[Tensor, Optimizer], None]:
    def update_fn(loss: Tensor, optimizer: Optimizer) -> None:
        grad_scaler = trainer.grad_scaler
        grad_scaler.scale(loss).backward()
        trainer.clip_norm_step()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()

    return update_fn


def run_train_steps(
    m: CustomModule,
    train_steps: List[CustomTrainStep],
    *,
    batch_idx: int,
    batch: tensor_dict_type,
    trainer: ITrainer,
    forward_kwargs: Dict[str, Any],
    loss_kwargs: Dict[str, Any],
) -> StepOutputs:
    state = trainer.state
    forward: Union[tensor_dict_type, List[tensor_dict_type]] = {}
    loss_dict = {}
    update_fn = get_update_fn(trainer)
    performed_scheduler_step = False
    get_fw = lambda: _forward(m, batch_idx, batch, INPUT_KEY, state, **forward_kwargs)
    for i, train_step in enumerate(train_steps):
        if train_step.should_skip(m, trainer.state):
            continue
        if i == 0 or train_step.requires_new_forward:
            if train_step.num_forward == 1:
                forward = get_fw()
            else:
                forward = [get_fw() for _ in range(train_step.num_forward)]
        optimizer = trainer.optimizers[train_step.scope]
        with toggle_optimizer(m, optimizer, enabled=train_step.enable_toggle_optimizer):
            with autocast(enabled=trainer.use_amp):
                loss_res = train_step.loss_fn(m, trainer, batch, forward, **loss_kwargs)
            update_fn(loss_res.loss, optimizer)
            loss_dict.update(loss_res.losses)
        performed_scheduler_step = train_step.requires_scheduler_step
        if performed_scheduler_step:
            trainer.scheduler_step()
    if not performed_scheduler_step:
        trainer.scheduler_step()
    return StepOutputs(forward, loss_dict)


def register_custom_module(
    name: str,
    *,
    allow_duplicate: bool = False,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
    custom_train_step: bool = True,
    custom_evaluate_step: bool = True,
    custom_params_groups: bool = False,
    custom_ddp_initialization: bool = False,
) -> Callable[[Type[CustomModule]], Type[CustomModule]]:
    def _core(m: Type[CustomModule]) -> Type[CustomModule]:
        @IDLModel.register(name, allow_duplicate=allow_duplicate)
        class _(*bases):  # type: ignore
            core: CustomModule

            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.core = m(*args, **kwargs)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                return _forward(self.core, batch_idx, batch, INPUT_KEY, state, **kwargs)

            def onnx_forward(self, batch: tensor_dict_type) -> Any:
                return self.core.onnx_forward(batch)

            def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
                self.core.summary_forward(batch_idx, batch)

            def train_step(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                trainer: ITrainer,
                forward_kwargs: Dict[str, Any],
                loss_kwargs: Dict[str, Any],
            ) -> StepOutputs:
                train_steps = self.core.train_steps
                if train_steps is not None:
                    return run_train_steps(
                        self.core,
                        train_steps,
                        batch_idx=batch_idx,
                        batch=batch,
                        trainer=trainer,
                        forward_kwargs=forward_kwargs,
                        loss_kwargs=loss_kwargs,
                    )
                kwargs = dict(
                    batch_idx=batch_idx,
                    batch=batch,
                    state=trainer.state,
                    optimizers=trainer.optimizers,
                    use_amp=trainer.use_amp,
                    grad_scaler=trainer.grad_scaler,
                    clip_norm_fn=trainer.clip_norm_step,
                    update_fn=get_update_fn(trainer),
                    scheduler_step_fn=trainer.scheduler_step,
                    trainer=trainer,
                    forward_kwargs=forward_kwargs,
                    loss_kwargs=loss_kwargs,
                )
                return safe_execute(self.core.train_step, kwargs)

            def evaluate_step(  # type: ignore
                self,
                loader: IDataLoader,
                portion: float,
                trainer: ITrainer,
            ) -> MetricsOutputs:
                kw = dict(
                    state=trainer.state,
                    weighted_loss_score_fn=trainer.weighted_loss_score,
                    trainer=trainer,
                )
                fn = self.core.evaluate_step
                final_scores = []
                metric_values: Dict[str, List[float]] = {}
                for i, batch in enumerate(loader):
                    if i / len(loader) >= portion:
                        break
                    batch = to_device(batch, self.device)
                    i_kw = shallow_copy_dict(kw)
                    i_kw["batch_idx"] = i
                    i_kw["batch"] = batch
                    out: MetricsOutputs = safe_execute(fn, i_kw)
                    final_scores.append(out.final_score)
                    for k, v in out.metric_values.items():
                        metric_values.setdefault(k, []).append(v)
                return MetricsOutputs(
                    sum(final_scores) / len(final_scores),
                    {k: sum(v) / len(v) for k, v in metric_values.items()},
                )

            def params_groups(self, m_: nn.Module) -> Any:
                return self.core.params_groups(m_)

            def init_ddp(self) -> None:
                self.core.init_ddp()

            def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
                self.core.permute_trainer_config(trainer_config)

            def _init_with_trainer(self, trainer: ITrainer) -> None:
                self.core.init_with_trainer(trainer)

        _.custom_train_step = custom_train_step
        _.custom_evaluate_step = custom_evaluate_step
        _.custom_params_groups = custom_params_groups
        _.custom_ddp_initialization = custom_ddp_initialization
        return m

    bases = (pre_bases or []) + [ModelWithCustomSteps] + (post_bases or [])
    return _core


def register_loss_module(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Registered module should have forward method with one of the following formats:

    * forward(self, predictions, labels, **kwargs)
        * `predictions` / `labels` could have a different name
    * forward(self, predictions, labels, state, **kwargs)
        * `predictions` / `labels` could have a different name
        * `state` should be exactly the same name
    * forward(self, forward_results, batch, **kwargs)
        * `forward_results` / `batch` should all be exactly the same name
    * forward(self, forward_results, batch, state, **kwargs)
        * `forward_results` / `batch` / `state` should all be exactly the same name

    Types:
    * predictions / labels: Tensor
    * forward_results / batch: tensor_dict_type
    * state: Optional[TrainerState]

    Note that the order of the arguments should always keep the same
    """

    def _core(loss_base: Type[nn.Module]) -> Type[nn.Module]:
        @ILoss.register(name, allow_duplicate=allow_duplicate)
        class _(ILoss):  # type: ignore
            def __init__(self, reduction: str = "mean", **kwargs: Any):
                super().__init__(reduction, **kwargs)
                self.core = loss_base(**kwargs)

            def _core(
                self,
                forward_results: tensor_dict_type,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> losses_type:
                args: List[Any] = []
                fn = self.core.forward
                num_args = get_num_positional_args(fn)
                if check_requires(fn, "forward_results"):
                    args.append(forward_results)
                else:
                    args.append(forward_results[PREDICTIONS_KEY])
                if num_args >= 2:
                    if check_requires(fn, "batch"):
                        args.append(batch)
                    else:
                        args.append(batch.get(LABEL_KEY))
                    if check_requires(fn, "state"):
                        args.append(state)
                return self.core(*args, **kwargs)

        return loss_base

    return _core


def register_transform(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable[[transform_type], transform_type]:
    def _core(transform_base: transform_type) -> transform_type:
        @Transforms.register(name, allow_duplicate=allow_duplicate)
        class _(Transforms):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.fn = transform_base(*args, **kwargs).forward

            @property
            def need_batch_process(self) -> bool:
                return True

            @property
            def need_numpy(self) -> bool:
                return False

        return transform_base

    return _core


__all__ = [
    "register_initializer",
    "register_metric",
    "Initializer",
    "register_module",
    "register_custom_module",
    "register_loss_module",
    "register_transform",
    "CustomTrainStep",
    "CustomTrainStepLoss",
    "CustomModule",
    "IMetric",
    "ITransform",
]
