import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from .protocol import MLModelProtocol
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY
from ...modules.blocks import Mapping


@MLModelProtocol.register("fcnn")
class FCNN(MLModelProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_units: Optional[List[int]] = None,
        *,
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(in_dim, out_dim)
        if hidden_units is None:
            dim = min(1024, 2 * in_dim)
            hidden_units = 2 * [dim]
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            mapping = Mapping(
                in_dim,
                hidden_unit,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
            blocks.append(mapping)
            in_dim = hidden_unit
        blocks.append(nn.Linear(in_dim, out_dim, bias))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.net(batch[INPUT_KEY])}


__all__ = ["FCNN"]
