import torch

from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .common import CommonMLModel
from .common import register_ml_model
from ...schema import forward_results_type
from ...schema import ITrainer
from ...schema import TrainerState
from ...modules import DDR
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY


@register_ml_model("ddr")
class DDRModel(CommonMLModel):
    def init_with_trainer(self, trainer: ITrainer) -> None:
        m: DDR = self.get_module()
        if m._y_min_max is None:
            y_train = trainer.train_loader.get_full_batch()[LABEL_KEY]
            m._y_min_max = y_train.min().item(), y_train.max().item()
        m.y_min_max = torch.tensor(m._y_min_max)

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return batch[INPUT_KEY], state

    def forward(  # type: ignore
        self,
        net: Tensor,
        state: Optional[TrainerState],
        **kwargs: Any,
    ) -> forward_results_type:
        net = self.encode(net).merged_all
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.get_module()(net, state, **kwargs)


__all__ = [
    "DDRModel",
]
