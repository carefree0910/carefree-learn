import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from .schema import EncoderMixin
from ....modules.blocks import make_attention
from ....modules.blocks import HijackConv2d
from ....modules.blocks import ResidualBlockWithTimeEmbedding


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        if not with_conv:
            self.conv = None
        else:
            self.conv = HijackConv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, net: Tensor) -> Tensor:
        if self.conv is None:
            return F.avg_pool2d(net, kernel_size=2, stride=2)
        pad = (0, 1, 0, 1)
        net = F.pad(net, pad, mode="constant", value=0)
        net = self.conv(net)
        return net


@EncoderMixin.register("attention")
class AttentionEncoder(nn.Module, EncoderMixin):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        downsample_with_conv: bool = True,
        attention_type: str = "spatial",
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.latent_channels = latent_channels
        self.num_downsample = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        make_res_block = lambda in_c, out_c: ResidualBlockWithTimeEmbedding(
            in_c,
            out_c,
            dropout=dropout,
            time_embedding_channels=0,
        )
        # in conv
        blocks = [HijackConv2d(in_channels, inner_channels, 3, 1, 1)]
        # downsample
        current_resolution = img_size
        in_channel_multipliers = (1,) + tuple(channel_multipliers)
        self.in_channel_multipliers = in_channel_multipliers
        in_nc = 0
        for i in range(self.num_downsample):
            in_nc = inner_channels * in_channel_multipliers[i]
            out_nc = inner_channels * channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                blocks.append(make_res_block(in_nc, out_nc))
                in_nc = out_nc
                if current_resolution in attention_resolutions:
                    blocks.append(make_attention(in_nc, attention_type))
            if i != self.num_downsample - 1:
                blocks.append(Downsample(in_nc, downsample_with_conv))
                current_resolution //= 2
        # residual
        blocks += [
            make_res_block(in_nc, in_nc),
            make_attention(in_nc, attention_type),
            make_res_block(in_nc, in_nc),
        ]
        # output
        blocks += [
            nn.GroupNorm(num_groups=32, num_channels=in_nc, eps=1.0e-6, affine=True),
            nn.SiLU(),
            HijackConv2d(in_nc, latent_channels, 3, 1, 1),
        ]
        # construct
        self.encoder = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.encoder(net)


__all__ = [
    "AttentionEncoder",
]
