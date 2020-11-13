import torch

from typing import Any
from typing import Dict

from .base import HeadBase
from ..blocks import Linear as Lin


@HeadBase.register("linear")
class Linear(HeadBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        in_dim, out_dim = map(config.get, ["in_dim", "out_dim"])
        linear_config = config["linear_config"]
        self.linear = Lin(in_dim, out_dim, **linear_config)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.linear(net)


__all__ = ["Linear"]
