import torch

from typing import Any
from typing import Dict
from typing import Optional

from .base import HeadBase
from ...modules.blocks import DNDF


@HeadBase.register("dndf")
class DNDFHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: Optional[Dict[str, Any]],
    ):
        super().__init__()
        if config is None:
            self.dndf = None
        else:
            self.dndf = DNDF(in_dim, out_dim, **config)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if self.dndf is None:
            return net
        return self.dndf(net)


__all__ = ["DNDFHead"]
