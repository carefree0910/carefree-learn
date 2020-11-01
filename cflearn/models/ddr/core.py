import torch

import torch.nn as nn

from typing import Dict
from typing import Callable
from typing import Optional

from ...modules.blocks import MLP
from ...modules.blocks import Mapping
from ...modules.blocks import InvertibleBlock
from ...modules.blocks import PseudoInvertibleBlock


class DDRCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        y_min: float,
        y_max: float,
        to_latent: bool = True,
        num_blocks: Optional[int] = None,
        latent_dim: Optional[int] = None,
        transition_builder: Callable[[int], nn.Module] = None,
    ):
        super().__init__()
        self.y_min = y_min
        self.y_diff = y_max - y_min
        # to latent
        if not to_latent:
            latent_dim = in_dim
            self.to_latent = None
        else:
            if latent_dim is None:
                latent_dim = 512
            latent_cfg = {
                "bias": False,
                "dropout": 0.0,
                "batch_norm": False,
                "activation": None,
            }
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
            block = InvertibleBlock(latent_dim, transition_builder)
            self.blocks.append(block)
        self.num_blocks = num_blocks

    @property
    def q_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 2.0 * q - 1.0

    @property
    def q_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda q: 0.5 * (q + 1.0)

    @property
    def y_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y - self.y_min) / (0.5 * self.y_diff) - 1.0

    @property
    def y_inv_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda y: (y + 1.0) * (0.5 * self.y_diff) + self.y_min

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
                q_batch = self.q_fn(q_batch)
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
            q_net = q_latent
            for block in self.blocks:
                q_net = block(q_net)
            y = self.y_invertible.inverse(q_net - latent)
            y = self.y_inv_fn(y)
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
            y_batch = self.y_fn(y_batch)
            y_latent = latent + self.y_invertible(y_batch)
        # simulate cdf
        y_inverse = None
        if y_latent is None:
            q = None
        else:
            y_net = y_latent
            for i in range(self.num_blocks):
                y_net = self.blocks[self.num_blocks - i - 1].inverse(y_net)
            q = torch.tanh(self.q_invertible.inverse(y_net - latent))
            q = self.q_inv_fn(q)
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
