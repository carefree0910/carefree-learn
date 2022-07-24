import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Optional

from .protocol import register_ml_module
from ...modules.blocks import mapping_dict


@register_ml_module("fcnn")
class FCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        input_dim *= num_history
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]
        mapping_base = mapping_dict[mapping_type]
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            mapping = mapping_base(
                input_dim,
                hidden_unit,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
            blocks.append(mapping)
            input_dim = hidden_unit
        blocks.append(nn.Linear(input_dim, output_dim, bias))
        self.hidden_units = hidden_units
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.net(net)


__all__ = ["FCNN"]
