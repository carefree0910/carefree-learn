from torch import nn
from torch import Tensor

from .utils import register_ml_module


@register_ml_module("linear")
class LinearModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim, bias)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "LinearModule",
]