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
from ..protocol import _forward
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
                return _forward(self.core, batch_idx, batch, key, state, **kwargs)

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


def register_custom_loss_module(
    name: str,
    *,
    is_ml: bool,
    allow_duplicate: bool = False,
    pre_bases: Optional[List[type]] = None,
    post_bases: Optional[List[type]] = None,
) -> Callable[[custom_loss_module_type], custom_loss_module_type]:
    def _core(m: custom_loss_module_type) -> custom_loss_module_type:
        class _(*bases):  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__()
                self.core = m(*args, **kwargs)

            def train_step(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                trainer: Any,
                forward_kwargs: Dict[str, Any],
                loss_kwargs: Dict[str, Any],
            ) -> StepOutputs:
                with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                    kwargs = dict(
                        batch_idx=batch_idx,
                        batch=batch,
                        trainer=trainer,
                        forward_kwargs=forward_kwargs,
                        loss_kwargs=loss_kwargs,
                    )
                    fn = self.core.get_losses
                    forward_results, loss_dict = fn(**filter_kw(fn, kwargs))
                trainer.post_loss_step(loss_dict)
                return StepOutputs(
                    forward_results, {k: v.item() for k, v in loss_dict.items()}
                )

            def evaluate_step(
                self,
                loader: DataLoaderProtocol,
                portion: float,
                trainer: Any,
            ) -> MetricsOutputs:
                if trainer.metrics is not None:
                    outputs = trainer.inference.get_outputs(
                        loader,
                        portion=portion,
                        state=trainer.state,
                        metrics=trainer.metrics,
                        return_outputs=False,
                    )
                    assert outputs.metric_outputs is not None
                    return outputs.metric_outputs
                loss_items: Dict[str, List[float]] = {}
                for i, batch in enumerate(loader):
                    if i / len(loader) >= portion:
                        break
                    batch = to_device(batch, self.device)
                    kwargs = dict(
                        batch_idx=0,
                        batch=batch,
                        trainer=trainer,
                        forward_kwargs={},
                        loss_kwargs={},
                    )
                    fn = self.core.get_losses
                    _, losses = fn(**filter_kw(fn, kwargs))
                    for k, v in losses.items():
                        loss_items.setdefault(k, []).append(v.item())
                # gather
                mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
                score = trainer.weighted_loss_score(mean_loss_items)
                return MetricsOutputs(score, mean_loss_items)

            def forward(
                self,
                batch_idx: int,
                batch: tensor_dict_type,
                state: Optional[TrainerState] = None,
                **kwargs: Any,
            ) -> tensor_dict_type:
                key = MERGED_KEY if is_ml and MERGED_KEY in batch else INPUT_KEY
                return _forward(self.core, batch_idx, batch, key, state, **kwargs)

        if is_ml:
            register_ml_module(name, allow_duplicate=allow_duplicate)(_)
        else:
            ModelWithCustomSteps.register(name, allow_duplicate=allow_duplicate)(_)
        return m

    bases = (pre_bases or []) + [ModelWithCustomSteps] + (post_bases or [])
    return _core


__all__ = [
    "register_ml_module",
    "register_custom_loss_module",
]
