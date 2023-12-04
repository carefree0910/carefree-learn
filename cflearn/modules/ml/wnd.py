from torch import nn
from torch import Tensor
from typing import List
from typing import Optional

from .fcnn import FCNN
from ..common import register_module


@register_module("wnd")
class WideAndDeep(nn.Module):
    def __init__(
        self,
        wide_dim: int,
        deep_dim: int,
        output_dim: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(wide_dim, output_dim, bias=bias)
        self.fcnn = FCNN(
            deep_dim,
            output_dim,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, wide_net: Tensor, deep_net: Tensor) -> Tensor:
        wide_net = self.linear(wide_net)
        deep_net = self.fcnn(deep_net)
        return wide_net + deep_net


__all__ = [
    "WideAndDeep",
]
