from torch import Tensor
from typing import Any
from torch.nn import Module


class Residual(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        return net + self.module(net, **kwargs)


__all__ = [
    "Residual",
]
