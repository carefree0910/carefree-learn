import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import List
from typing import Type
from typing import Callable
from typing import Optional

from ...types import losses_type
from ...types import tensor_dict_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...protocol import ModelProtocol
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
                self.core = m(*args, **kwargs)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                rs = self.core(batch[INPUT_KEY], **kwargs)
                if isinstance(rs, Tensor):
                    rs = {PREDICTIONS_KEY: rs}
                return rs

        return m

    bases = (pre_bases or []) + [ModelProtocol] + (post_bases or [])
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
    "register_loss_module",
    "LossModule",
]
