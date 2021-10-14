import torch.nn as nn

from typing import Any
from typing import List
from typing import Type
from typing import Callable
from typing import Optional

from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...protocol import ModelProtocol
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
                return {PREDICTIONS_KEY: self.core(batch[INPUT_KEY], **kwargs)}

        return m

    bases = (pre_bases or []) + [ModelProtocol] + (post_bases or [])
    return _core


__all__ = [
    "register_module",
]
