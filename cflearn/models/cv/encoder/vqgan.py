import torch

import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Optional

from .schema import EncoderMixin
from ....modules.blocks import Conv2d


def normalize(dim: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=dim, eps=1.0e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.in_channels = dim
        self.norm = normalize(dim)
        self.q = Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.k = Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.v = Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.norm(net)
        q = self.q(net)
        k = self.k(net)
        v = self.v(net)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        weights = torch.bmm(q, k)
        weights = weights * (int(c) ** (-0.5))
        weights = torch.nn.functional.softmax(weights, dim=2)

        v = v.reshape(b, c, h * w)
        weights = weights.permute(0, 2, 1)
        net = torch.bmm(v, weights)
        net = net.reshape(b, c, h, w)
        net = self.proj_out(net)
        return inp + net


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)

    def forward(self, net: Tensor) -> Tensor:
        net = nn.functional.pad(net, (0, 1, 0, 1), mode="constant", value=0.0)
        net = self.net(net)
        return net


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            normalize(in_channels),
            nn.SiLU(),
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            normalize(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels == out_channels:
            self.shortcut = None
        else:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.net(net)
        if self.shortcut is not None:
            inp = self.shortcut(inp)
        return inp + net


@EncoderMixin.register("vqgan")
class VQGANEncoder(nn.Module, EncoderMixin):
    def __init__(
        self,
        img_size: int,
        in_channels: int = 3,
        latent_channels: int = 256,
        *,
        channel_multipliers: Optional[List[int]] = None,
        attention_resolutions: Optional[List[int]] = None,
        latent_dim: int = 128,
        res_dropout: float = 0.0,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 1, 2, 2, 4]
        if attention_resolutions is None:
            attention_resolutions = [16]
        num_downsample = len(channel_multipliers)
        self.in_channels = in_channels
        self.num_downsample = num_downsample
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        # in conv
        blocks = [Conv2d(in_channels, latent_dim, kernel_size=3, padding=1)]
        # downsample
        current_res = img_size
        in_channel_multipliers = [1] + channel_multipliers
        in_nc = 0
        for i in range(num_downsample):
            i_blocks = []
            in_nc = latent_dim * in_channel_multipliers[i]
            out_nc = latent_dim * channel_multipliers[i]
            for j in range(num_res_blocks):
                i_blocks.append(
                    ResidualBlock(
                        in_nc,
                        out_nc,
                        dropout=res_dropout,
                    )
                )
                in_nc = out_nc
                if current_res in attention_resolutions:
                    i_blocks.append(AttnBlock(in_nc))
            if i != num_downsample - 1:
                i_blocks.append(Downsample(in_nc))
                current_res //= 2
            blocks.append(nn.Sequential(*i_blocks))
        # mid
        blocks.extend(
            [
                ResidualBlock(in_nc, in_nc, res_dropout),
                AttnBlock(in_nc),
                ResidualBlock(in_nc, in_nc, res_dropout),
            ]
        )
        # finalize
        blocks.extend(
            [
                normalize(in_nc),
                nn.SiLU(),
                Conv2d(in_nc, latent_channels, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "normalize",
    "AttnBlock",
    "ResidualBlock",
    "VQGANEncoder",
]
