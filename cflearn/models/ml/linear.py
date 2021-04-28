import torch.nn as nn

from typing import Any
from typing import Optional

from .protocol import MERGED_KEY
from .protocol import MLCoreProtocol
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...constants import PREDICTIONS_KEY


@MLCoreProtocol.register("linear")
class Linear(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        *,
        bias: bool = True,
    ):
        super().__init__(in_dim, out_dim, num_history)
        in_dim *= num_history
        self.net = nn.Linear(in_dim, out_dim, bias)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[MERGED_KEY]
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return {PREDICTIONS_KEY: self.net(net)}


__all__ = ["Linear"]
