import torch

import torch.nn as nn

from typing import *

from .base import HeadBase
from ...modules.blocks import DNDF
from ...modules.blocks import TreeResBlock


@HeadBase.register("tree_stack")
class TreeStackHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        dndf_config: Dict[str, Any],
        out_dndf_config: Dict[str, Any],
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(TreeResBlock(in_dim, dndf_config))
        self.out_dndf = DNDF(
            in_dim,
            out_dim,
            **out_dndf_config,
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            net = block(net)
        return self.out_dndf(net)


__all__ = ["TreeStackHead"]
