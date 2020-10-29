import torch

import torch.nn as nn

from typing import *

from ...modules.blocks import Linear


class LinearCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        linear_config: Dict[str, Any],
    ):
        super().__init__()
        self.linear = Linear(in_dim, out_dim, **linear_config)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.linear(net)


__all__ = ["LinearCore"]
