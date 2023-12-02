import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Optional
from cftool.misc import safe_execute

from .utils import register_ml_module
from ..core.mappings import mappings


@register_ml_module("fcnn")
class FCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
    ):
        super().__init__()
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]
        mapping_base = mappings.get(mapping_type)
        if mapping_base is None:
            raise ValueError(f"cannot find mapping type: `{mapping_type}`")
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            mapping = safe_execute(
                mapping_base,
                dict(
                    in_dim=input_dim,
                    out_dim=hidden_unit,
                    bias=bias,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    rank=rank,
                    rank_ratio=rank_ratio,
                ),
            )
            blocks.append(mapping)
            input_dim = hidden_unit
        blocks.append(nn.Linear(input_dim, output_dim, bias))
        self.hidden_units = hidden_units
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "FCNN",
]
