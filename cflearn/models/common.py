import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from cftool.misc import safe_execute
from cftool.types import tensor_dict_type

from ..losses import build_loss
from ..schema import ILoss
from ..schema import DLConfig
from ..schema import IDLModel
from ..schema import TrainStep
from ..schema import TrainerState
from ..schema import TrainStepLoss
from ..modules import build_module
from ..toolkit import get_clones
from ..constants import LOSS_KEY


class CommonTrainStep(TrainStep):
    def __init__(self, loss: ILoss):
        super().__init__()
        self.loss = loss

    def loss_fn(
        self,
        m: IDLModel,
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        losses = self.loss.run(forward_results, batch, state)
        return TrainStepLoss(
            losses[LOSS_KEY],
            {k: v.item() for k, v in losses.items()},
        )


@IDLModel.register("common")
class CommonDLModel(IDLModel):
    loss: ILoss

    @property
    def train_steps(self) -> List[TrainStep]:
        return [CommonTrainStep(self.loss)]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m, self.loss]

    def build(self, config: DLConfig) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `CommonDLModel`")
        self.m = build_module(config.module_name, config=config.module_config)
        self.loss = build_loss(config.loss_name, config=config.loss_config)


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class DLEnsembleModel(CommonDLModel):
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IDLModel, num_repeat: int) -> None:
        super().__init__()
        self.m = get_clones(m.m, num_repeat)
        self.model = m
        self.config = m.config.copy()
        self.ensemble_fn = None
        self.__identifier__ = m.__identifier__

    @property
    def train_steps(self) -> List[TrainStep]:
        return self.model.train_steps

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m]

    def build(self, config: DLConfig) -> None:
        raise ValueError("`build` should not be called for `DLEnsembleModel`")

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        outputs: Dict[str, List[Tensor]] = {}
        for m in self.m:
            self.model.m = m
            m_outputs = self.model.run(batch_idx, batch, state, **kwargs)
            for k, v in m_outputs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results


__all__ = [
    "CommonTrainStep",
    "CommonDLModel",
    "DLEnsembleModel",
]
