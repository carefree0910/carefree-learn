import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict

from ...modules.blocks import Mapping
from ...modules.blocks import InvertibleBlock


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_blocks: int = 3,
        to_latent: bool = True,
        latent_dim: Optional[int] = None,
        num_units: Optional[List[int]] = None,
        mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        latent_cfg = {
            "bias": False,
            "dropout": 0.0,
            "batch_norm": False,
            "activation": "mish",
        }
        # to latent
        if not to_latent:
            latent_dim = in_dim
            self.to_latent = None
        else:
            if latent_dim is None:
                latent_dim = 256
            self.to_latent = Mapping(in_dim, latent_dim, **latent_cfg)  # type: ignore
        # invertible blocks
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
        median: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # prepare latent features
        latent = net if self.to_latent is None else self.to_latent(net)
        if not median:
            if q_batch is None:
                q_latent = None
            else:
                q_latent = latent + torch.atanh(q_batch)
        else:
            if q_batch is not None:
                msg = "`median` is specified but `q_batch` is still provided"
                raise ValueError(msg)
            q_latent = latent
        if y_batch is None:
            y_latent = None
        else:
            y_latent = latent + y_batch
        # simulate quantile function
        if q_latent is None:
            y_reduced = y_predictions = None
        else:
            q1, q2 = q_latent.chunk(2, dim=1)
            for block in self.blocks:
                q1, q2 = block(q1, q2)
            q_final = torch.cat([q1, q2], dim=1)
            y_predictions = q_final - latent
            y_reduced = y_predictions.mean(1, keepdims=True)
        # simulate cdf
        if y_latent is None:
            q_reduced = q_predictions = None
        else:
            y1, y2 = y_latent.chunk(2, dim=1)
            for i in range(self.num_blocks):
                y1, y2 = self.blocks[self.num_blocks - i - 1].inverse(y1, y2)
            y_final = torch.cat([y1, y2], dim=1)
            q_predictions = 0.5 * ((y_final - latent).tanh() + 1.0)
            q_reduced = q_predictions.mean(1, keepdims=True)
        return {
            "q": q_reduced,
            "y": y_reduced,
            "q_full": q_predictions,
            "y_full": y_predictions,
        }


__all__ = ["DDRCore"]
