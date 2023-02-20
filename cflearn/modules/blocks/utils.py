import torch.nn as nn

from torch import Tensor
from typing import Any
from torch.nn import Module

from .convs import HijackConv1d
from .convs import HijackConv2d
from .convs import HijackConv3d


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


def conv_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return HijackConv1d(*args, **kwargs)
    elif n == 2:
        return HijackConv2d(*args, **kwargs)
    elif n == 3:
        return HijackConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


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
    "conv_nd",
    "avg_pool_nd",
    "zero_module",
]
