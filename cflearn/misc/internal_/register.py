import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional
from torch.cuda.amp import GradScaler
from torch.optim.optimizer import Optimizer

from ...types import losses_type
from ...types import tensor_dict_type
from ...protocol import StepOutputs
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...protocol import ModelProtocol
from ...protocol import MetricsOutputs
from ...protocol import DataLoaderProtocol
from ...protocol import ModelWithCustomSteps
from ...constants import LOSS_KEY
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY


def register_module(
    name: str,
    *,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _core(m: Type[nn.Module]) -> Type[nn.Module]:
        @ModelProtocol.register(name)
        class _(*bases):  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.net = m(*args, **kwargs)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                rs = self.net(batch[INPUT_KEY], **kwargs)
                if not isinstance(rs, dict):
                    rs = {PREDICTIONS_KEY: rs}
                return rs

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

    @staticmethod
    def params_groups(m: nn.Module) -> Any:
        pass

    def init_ddp(self) -> None:
        pass

    def permute_trainer_config(self, trainer_config: Dict[str, Any]) -> None:
        pass


def register_custom_module(
    name: str,
    *,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
    custom_train_step: bool = True,
    custom_evaluate_step: bool = True,
    custom_params_groups: bool = False,
    custom_ddp_initialization: bool = False,
) -> Callable[[Type[CustomModule]], Type[CustomModule]]:
    def _core(m: Type[CustomModule]) -> Type[CustomModule]:
        @ModelProtocol.register(name)
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
                return self.core.forward(batch_idx, batch, state, **kwargs)

            def train_step(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                trainer: Any,
                forward_kwargs: Dict[str, Any],
                loss_kwargs: Dict[str, Any],
            ) -> StepOutputs:
                return self.core.train_step(
                    batch_idx,
                    batch,
                    trainer.optimizers,
                    trainer.use_amp,
                    trainer.grad_scaler,
                    _get_clip_norm_fn(trainer),
                    trainer.scheduler_step,
                    trainer,
                    forward_kwargs,
                    loss_kwargs,
                )

            def evaluate_step(  # type: ignore
                self,
                loader: DataLoaderProtocol,
                portion: float,
                trainer: Any,
            ) -> MetricsOutputs:
                return self.core.evaluate_step(
                    loader,
                    portion,
                    trainer.weighted_loss_score,
                    trainer,
                )

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


class LossModule(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        pass


def register_loss_module(name: str) -> Callable[[Type[LossModule]], Type[LossModule]]:
    def _core(loss_base: Type[LossModule]) -> Type[LossModule]:
        @LossProtocol.register(name)
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
                rs = self.core(forward_results, batch, state, **kwargs)
                if isinstance(rs, Tensor):
                    rs = {LOSS_KEY: rs}
                return rs

        return loss_base

    return _core


__all__ = [
    "register_module",
    "register_custom_module",
    "register_loss_module",
    "CustomModule",
    "LossModule",
]
