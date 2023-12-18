from abc import abstractmethod
from abc import ABCMeta
from torch import nn
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from cftool.array import tensor_dict_type

from ..schema import losses_type
from ..schema import ILoss
from ..schema import TrainerState
from ..constants import LOSS_KEY
from ..constants import PREDICTIONS_KEY
from ..modules.common import PrefixModules


TLoss = TypeVar("TLoss", bound=Type[ILoss])

losses = PrefixModules("loss")


def register_loss(name: str, **kwargs: Any) -> Callable[[TLoss], TLoss]:
    def before_register(cls: TLoss) -> TLoss:
        cls.__identifier__ = name
        return cls

    kwargs.setdefault("before_register", before_register)
    return losses.register(name, **kwargs)


def build_loss(
    name: str,
    *,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ILoss:
    return losses.build(name, config=config, **kwargs)


class _MultiLoss(ILoss, metaclass=ABCMeta):
    loss_weights: Dict[str, float]

    def __init__(
        self,
        reduction: str = "mean",
        *,
        loss_names: List[str],
        loss_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(reduction)
        configs = loss_configs or {}
        weights = loss_weights or {}
        loss_list = [build_loss(n, config=configs.get(n, {})) for n in loss_names]
        self.base_losses = nn.ModuleList(loss_list)
        self.init_loss_weights(loss_names, weights)

    @abstractmethod
    def init_loss_weights(self, names: List[str], weights: Dict[str, float]) -> None:
        pass

    @staticmethod
    def _inject(key: str, new_losses: losses_type, all_losses: losses_type) -> None:
        if isinstance(new_losses, dict):
            new_losses = new_losses[LOSS_KEY]
        all_losses[key] = new_losses

    def _merge(self, losses: tensor_dict_type) -> None:
        merged = 0.0
        for k, v in self.loss_weights.items():
            k_loss = losses.get(k)
            if k_loss is not None:
                merged += k_loss * v
        losses[LOSS_KEY] = merged


@register_loss("multi_task")
class MultiTaskLoss(_MultiLoss):
    def init_loss_weights(self, names: List[str], weights: Dict[str, float]) -> None:
        if len(names) != len(set(names)):
            raise ValueError("`loss_names` should be unique")
        self.loss_weights = {k: weights.get(k, 1.0) for k in names}

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
    def init_loss_weights(self, names: List[str], weights: Dict[str, float]) -> None:
        loss_weights = {f"{i}_{k}": weights.get(k, 1.0) for i, k in enumerate(names)}
        self.loss_weights = loss_weights

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
