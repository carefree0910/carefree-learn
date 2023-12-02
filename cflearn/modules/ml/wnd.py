from torch import nn
from torch import Tensor
from typing import List
from typing import Optional

from .utils import register_ml_module
from .fcnn import FCNN


@register_ml_module("wnd")
class WideAndDeep(nn.Module):
    def __init__(
        self,
        deep_dim: int,
        wide_dim: Optional[int],
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
        if wide_dim is None:
            self.linear = None
        else:
            self.linear = nn.Linear(wide_dim, output_dim, bias=bias)

    def forward(self, deep_net: Tensor, wide_net: Optional[Tensor] = None) -> Tensor:
        deep_net = self.fcnn(deep_net)
        if wide_net is None:
            if self.linear is not None:
                raise ValueError("`wide_net` is required since `wide_dim` is provided")
            return deep_net
        if self.linear is None:
            raise ValueError("`wide_net` is provided but `wide_dim` is not")
        wide_net = self.linear(wide_net)
        return deep_net + wide_net


__all__ = [
    "WideAndDeep",
]
