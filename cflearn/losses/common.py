from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Callable
from typing import Optional
from cftool.array import tensor_dict_type

from ..schema import losses_type
from ..schema import ILoss
from ..schema import TrainerState
from ..constants import LOSS_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.common import PrefixModules


TLoss = Type[ILoss]

losses = PrefixModules("loss")


def register_loss(name: str, **kwargs: Any) -> Callable[[TLoss], TLoss]:
    return losses.register(name, **kwargs)


def build_loss(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> ILoss:
    return losses.build(name, config=config, **kwargs)


class _MultiLoss(ILoss):
    def __init__(
        self,
        reduction: str = "mean",
        *,
        loss_names: List[str],
        loss_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(reduction)
        loss_configs = loss_configs or {}
        loss_weights = loss_weights or {}
        self.loss_weights = {k: loss_weights.get(k, 1.0) for k in loss_names}
        self.base_losses = nn.ModuleList(
            [build_loss(name, config=loss_configs.get(name, {})) for name in loss_names]
        )

    @staticmethod
    def _inject(key: str, base_losses: losses_type, all_losses: losses_type) -> None:
        if isinstance(base_losses, dict):
            base_losses = base_losses[LOSS_KEY]
        all_losses[key] = base_losses

    def _merge(self, losses: tensor_dict_type) -> None:
        merged = 0.0
        for k, v in self.loss_weights.items():
            k_loss = losses.get(k)
            if k_loss is not None:
                merged += k_loss * v
        losses[LOSS_KEY] = merged


@register_loss("multi_task")
class MultiTaskLoss(_MultiLoss):
    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        losses: losses_type = {}
        for loss_ins in self.base_losses:
            self._inject(
                loss_ins.__identifier__,
                loss_ins.run(forward_results, batch, state),
                losses,
            )
        self._merge(losses)
        return losses


@register_loss("multi_stage")
class MultiStageLoss(_MultiLoss):
    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        predictions = forward_results[PREDICTIONS_KEY]
        losses: tensor_dict_type = {}
        for i, pred in enumerate(predictions):
            forward_results[PREDICTIONS_KEY] = pred
            for loss_ins in self.base_losses:
                self._inject(
                    f"{i}_{loss_ins.__identifier__}",
                    loss_ins.run(forward_results, batch, state),
                    losses,
                )
        self._merge(losses)
        return losses


__all__ = [
    "losses",
    "register_loss",
    "build_loss",
    "MultiTaskLoss",
    "MultiStageLoss",
]
