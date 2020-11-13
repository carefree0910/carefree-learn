import torch

from typing import Any
from typing import Dict
from typing import Optional

from .base import HeadBase
from ..blocks import Linear as Lin


@HeadBase.register("linear")
class LinearHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        linear_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if linear_config is None:
            linear_config = {}
        self.linear = Lin(in_dim, out_dim, **linear_config)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.linear(net)


__all__ = ["LinearHead"]
