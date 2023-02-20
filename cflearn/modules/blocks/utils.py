import torch.nn as nn

from torch import Tensor
from typing import Any
from torch.nn import Module


class Residual(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        return net + self.module(net, **kwargs)


def zero_module(module: Module) -> Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif n == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif n == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


__all__ = [
    "Residual",
    "avg_pool_nd",
    "zero_module",
]
