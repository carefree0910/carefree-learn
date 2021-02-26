import torch
import cflearn

from typing import Any
from cflearn.modules.blocks import Linear


class ExternalLinear(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **_: Any):
        super().__init__()
        self.linear = Linear(in_dim, out_dim)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.linear(net)


cflearn.register_module("external_linear", ExternalLinear)
