import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional
from cftool.misc import check_requires
from torch.cuda.amp import GradScaler
from torch.optim.optimizer import Optimizer

from ..toolkit import Initializer
from ...types import losses_type
from ...types import np_dict_type
from ...types import tensor_dict_type
from ...protocol import StepOutputs
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...protocol import ModelProtocol
from ...protocol import MetricProtocol
from ...protocol import MetricsOutputs
from ...protocol import DataLoaderProtocol
from ...protocol import ModelWithCustomSteps
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY
from ...data.core import Transforms
from ...misc.toolkit import filter_kw
from ...misc.toolkit import shallow_copy_dict


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


class MetricInterface:
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


metric_type = Type[MetricInterface]


def register_metric(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable[[metric_type], metric_type]:
    def _core(metric_base: metric_type) -> metric_type:
        @MetricProtocol.register(name, allow_duplicate=allow_duplicate)
        class _(MetricProtocol):
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
                loader: Optional[DataLoaderProtocol],
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


def _forward(
    self: Any,
    batch_idx: int,
    batch: tensor_dict_type,
    general_input_key: str,
    state: Optional[TrainerState] = None,
    **kwargs: Any,
) -> tensor_dict_type:
    fn = self.core.forward
    kw = filter_kw(fn, kwargs)
    args: List[Any] = []
    if check_requires(fn, "batch_idx"):
        args.append(batch_idx)
    args.append(batch if check_requires(fn, "batch") else batch[general_input_key])
    if check_requires(fn, "state"):
        args.append(state)
    rs = self.core(*args, **kw)
    if not isinstance(rs, dict):
        rs = {PREDICTIONS_KEY: rs}
    return rs


def register_module(
    name: str,
    *,
    allow_duplicate: bool = False,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _core(m: Type[nn.Module]) -> Type[nn.Module]:
        @ModelProtocol.register(name, allow_duplicate=allow_duplicate)
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
                return _forward(self, batch_idx, batch, INPUT_KEY, state, **kwargs)

        return m

    bases = (pre_bases or []) + [ModelProtocol] + (post_bases or [])
    return _core


def _get_clip_norm_fn(trainer: Any) -> Callable[[], None]:
    def _core() -> None:
        if trainer.clip_norm > 0.0:
            trainer.clip_norm_step()

    return _core


class CustomModule(nn.Module, metaclass=ABCMeta):
    @property
    def device(self) -> torch.device:
        return list(self.parameters())[0].device

    @abstractmethod
    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        optimizers: Dict[str, Optimizer],
        use_amp: bool,
        grad_scaler: GradScaler,
        clip_norm_fn: Callable[[], None],
        scheduler_step_fn: Callable[[], None],
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
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: Any,
    ) -> MetricsOutputs:
        pass

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def params_groups(self, m: nn.Module) -> Any:
        pass

    def init_ddp(self) -> None:
        pass

    def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        pass


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
        @ModelProtocol.register(name, allow_duplicate=allow_duplicate)
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
                return _forward(self, batch_idx, batch, INPUT_KEY, state, **kwargs)

            def train_step(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                trainer: Any,
                forward_kwargs: Dict[str, Any],
                loss_kwargs: Dict[str, Any],
            ) -> StepOutputs:
                kwargs = dict(
                    batch_idx=batch_idx,
                    batch=batch,
                    optimizers=trainer.optimizers,
                    use_amp=trainer.use_amp,
                    grad_scaler=trainer.grad_scaler,
                    clip_norm_fn=_get_clip_norm_fn(trainer),
                    scheduler_step_fn=trainer.scheduler_step,
                    trainer=trainer,
                    forward_kwargs=forward_kwargs,
                    loss_kwargs=loss_kwargs,
                )
                fn = self.core.train_step
                return fn(**filter_kw(fn, kwargs))

            def evaluate_step(  # type: ignore
                self,
                loader: DataLoaderProtocol,
                portion: float,
                trainer: Any,
            ) -> MetricsOutputs:
                kwargs = dict(
                    loader=loader,
                    portion=portion,
                    weighted_loss_score_fn=trainer.weighted_loss_score,
                    trainer=trainer,
                )
                fn = self.core.evaluate_step
                return fn(**filter_kw(fn, kwargs))

            def params_groups(self, m_: nn.Module) -> Any:
                return self.core.params_groups(m_)

            def init_ddp(self) -> None:
                self.core.init_ddp()

            def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
                self.core.permute_trainer_config(trainer_config)

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
        @LossProtocol.register(name, allow_duplicate=allow_duplicate)
        class _(LossProtocol):  # type: ignore
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
                if check_requires(fn, "forward_results"):
                    args.append(forward_results)
                else:
                    args.append(forward_results[PREDICTIONS_KEY])
                if check_requires(fn, "batch"):
                    args.append(batch)
                else:
                    args.append(batch[LABEL_KEY])
                if check_requires(fn, "state"):
                    args.append(state)
                return self.core(*args, **kwargs)

        return loss_base

    return _core


class TransformInterface:
    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> tensor_dict_type:
        pass


transform_type = Type[TransformInterface]


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
    "CustomModule",
    "MetricInterface",
    "TransformInterface",
]
