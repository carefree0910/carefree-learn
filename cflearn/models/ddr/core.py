import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ...modules.blocks import Mapping
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import PseudoInvertibleBlock


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        to_latent: bool = True,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_units: Optional[List[int]] = None,
        mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        latent_cfg = {
            "bias": False,
            "dropout": 0.0,
            "batch_norm": False,
            "activation": None,
        }
        # to latent
        if not to_latent:
            latent_dim = in_dim
            self.to_latent = None
        else:
            if latent_dim is None:
                latent_dim = 512
            self.to_latent = Mapping(in_dim, latent_dim, **latent_cfg)  # type: ignore
        # pseudo invertible q / y
        self.q_invertible = PseudoInvertibleBlock(1, latent_dim)
        self.y_invertible = PseudoInvertibleBlock(1, latent_dim)
        # invertible blocks
        if num_blocks is None:
            num_blocks = 4
        if num_blocks % 2 != 0:
            raise ValueError("`num_blocks` should be divided by 2")
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = InvertibleBlock(latent_dim, num_units, mapping_config)  # type: ignore
            self.blocks.append(block)
        self.num_blocks = num_blocks

    def forward(
        self,
        net: torch.Tensor,
        *,
        q_batch: Optional[torch.Tensor] = None,
        y_batch: Optional[torch.Tensor] = None,
        do_inverse: bool = False,
        is_latent: bool = False,
        median: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if is_latent:
            latent = net
        else:
            latent = net if self.to_latent is None else self.to_latent(net)
        # prepare q_latent
        if not median:
            if q_batch is None:
                q_latent = None
            else:
                q_latent = latent + self.q_invertible(q_batch)
        else:
            if q_batch is not None:
                msg = "`median` is specified but `q_batch` is still provided"
                raise ValueError(msg)
            q_latent = latent
        # simulate quantile function
        q_inverse = None
        if q_latent is None:
            y = None
        else:
            q1, q2 = q_latent.chunk(2, dim=1)
            for block in self.blocks:
                q1, q2 = block(q1, q2)
            q_final = torch.cat([q1, q2], dim=1)
            y = self.y_invertible.inverse(q_final - latent)
            if do_inverse:
                q_inverse = self.forward(
                    latent,
                    y_batch=y,
                    is_latent=True,
                    do_inverse=False,
                )["q"]
        # prepare y_latent
        if y_batch is None:
            y_latent = None
        else:
            y_latent = latent + self.y_invertible(y_batch)
        # simulate cdf
        y_inverse = None
        if y_latent is None:
            q = None
        else:
            y1, y2 = y_latent.chunk(2, dim=1)
            for i in range(self.num_blocks):
                y1, y2 = self.blocks[self.num_blocks - i - 1].inverse(y1, y2)
            y_final = torch.cat([y1, y2], dim=1)
            q = torch.tanh(self.q_invertible.inverse(y_final - latent))
            if do_inverse:
                y_inverse = self.forward(
                    latent,
                    q_batch=q,
                    is_latent=True,
                    do_inverse=False,
                )["y"]
        return {
            "q": q,
            "y": y,
            "q_inverse": q_inverse,
            "y_inverse": y_inverse,
        }


__all__ = ["DDRCore"]
