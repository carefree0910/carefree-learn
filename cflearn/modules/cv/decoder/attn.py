import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple
from typing import Optional

from ..common import register_decoder
from ..common import IDecoder
from ...core import make_attention
from ...core import HijackConv2d
from ...core import ResidualBlockWithTimeEmbedding


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        if not with_conv:
            self.conv = None
        else:
            self.conv = HijackConv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, net: Tensor) -> Tensor:
        net = F.interpolate(net, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            net = self.conv(net)
        return net


@register_decoder("attention")
class AttentionDecoder(IDecoder):
    def __init__(
        self,
        img_size: int,
        out_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        upsample_with_conv: bool = True,
        attention_type: str = "spatial",
        apply_tanh: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.latent_channels = latent_channels
        self.num_upsample = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.apply_tanh = apply_tanh
        make_res_block = lambda in_c, out_c: ResidualBlockWithTimeEmbedding(
            in_c,
            out_c,
            dropout=dropout,
            time_embedding_channels=0,
        )
        # in conv
        in_nc = inner_channels * channel_multipliers[-1]
        blocks = [HijackConv2d(latent_channels, in_nc, 3, 1, 1)]
        # residual
        blocks += [
            make_res_block(in_nc, in_nc),
            make_attention(in_nc, attention_type),
            make_res_block(in_nc, in_nc),
        ]
        # upsample
        current_resolution = img_size // 2 ** (self.num_upsample - 1)
        for i in reversed(range(self.num_upsample)):
            out_nc = inner_channels * channel_multipliers[i]
            for _ in range(self.num_res_blocks + 1):
                blocks.append(make_res_block(in_nc, out_nc))
                in_nc = out_nc
                if current_resolution in attention_resolutions:
                    blocks.append(make_attention(in_nc, attention_type))
            if i != 0:
                blocks.append(Upsample(in_nc, upsample_with_conv))
                current_resolution *= 2
        # head
        head_blocks = [
            nn.GroupNorm(num_groups=32, num_channels=in_nc, eps=1.0e-6, affine=True),
            nn.SiLU(),
            HijackConv2d(in_nc, out_channels, 3, 1, 1),
        ]
        # construct
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(*head_blocks)

    def forward(
        self,
        net: Tensor,
        *,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        net = self.decoder(net)
        if no_head:
            return net
        net = self.head(net)
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        if apply_tanh:
            net = torch.tanh(net)
        return net

    def decode(
        self,
        net: Tensor,
        *,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        return self(net, no_head=no_head, apply_tanh=apply_tanh)


__all__ = [
    "AttentionDecoder",
]
