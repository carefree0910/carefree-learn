import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional

from .bases import custom_loss_module_type
from ..types import tensor_dict_type
from ..protocol import StepOutputs
from ..protocol import TrainerState
from ..protocol import MetricsOutputs
from ..protocol import DataLoaderProtocol
from ..protocol import ModelWithCustomSteps
from ..constants import INPUT_KEY
from .protocols.ml import MERGED_KEY
from .protocols.ml import MLCoreProtocol
from ..misc.toolkit import filter_kw
from ..misc.toolkit import to_device
from ..misc.internal_.register import _forward


def register_ml_module(
    name: str,
    *,
    allow_duplicate: bool = False,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _core(m: Type[nn.Module]) -> Type[nn.Module]:
        @MLCoreProtocol.register(name, allow_duplicate=allow_duplicate)
        class _(*bases):  # type: ignore
            def __init__(self, **kwargs: Any):
                super().__init__(**filter_kw(MLCoreProtocol.__init__, kwargs))
                kwargs["in_dim"] = kwargs["input_dim"]
                kwargs["out_dim"] = kwargs["output_dim"]
                self.core = m(**filter_kw(m, kwargs))

            def _init_with_trainer(self, trainer: Any) -> None:
                init_fn = getattr(self.core, "_init_with_trainer", None)
                if init_fn is not None:
                    init_fn(trainer)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                key = MERGED_KEY if MERGED_KEY in batch else INPUT_KEY
                return _forward(self, batch_idx, batch, key, state, **kwargs)

            def train_step(  # type: ignore
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                trainer: Any,
                forward_kwargs: Dict[str, Any],
                loss_kwargs: Dict[str, Any],
            ) -> StepOutputs:
                train_step = getattr(self.core, "train_step", None)
                if train_step is not None:
                    return train_step(
                        batch_idx,
                        batch,
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
                evaluate_step = getattr(self.core, "evaluate_step", None)
                if evaluate_step is not None:
                    return evaluate_step(loader, portion, trainer)

        return m

    bases = (pre_bases or []) + [MLCoreProtocol] + (post_bases or [])
    return _core


__all__ = [
    "register_ml_module",
]
