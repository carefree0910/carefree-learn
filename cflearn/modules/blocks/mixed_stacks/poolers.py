import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List
from typing import Optional
from torch.nn import Module

from ..customs import Linear


class BertPooler(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, net: Tensor) -> Tensor:
        net = net[:, 0]
        net = self.linear(net)
        net = self.activation(net)
        return net


class SequencePooler(Module):
    def __init__(self, dim: int, aux_heads: Optional[List[str]], bias: bool = True):
        super().__init__()
        self.out_dim = 1 + (0 if aux_heads is None else len(aux_heads))
        self.projection = Linear(dim, self.out_dim, bias=bias)

    def forward(self, net: Tensor) -> Tensor:
        weights = self.projection(net)
        weights = F.softmax(weights, dim=1).transpose(-1, -2)
        net = torch.matmul(weights, net)
        if self.out_dim > 1:
            return net
        return net.squeeze(-2)


__all__ = [
    "BertPooler",
    "SequencePooler",
]
