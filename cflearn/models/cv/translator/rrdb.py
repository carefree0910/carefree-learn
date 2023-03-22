import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from ....schema import IDLModel


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels: int, grow_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, grow_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + grow_channels, grow_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * grow_channels, grow_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * grow_channels, grow_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * grow_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net1 = self.lrelu(self.conv1(inp))
        net2 = self.lrelu(self.conv2(torch.cat((inp, net1), 1)))
        net3 = self.lrelu(self.conv3(torch.cat((inp, net1, net2), 1)))
        net4 = self.lrelu(self.conv4(torch.cat((inp, net1, net2, net3), 1)))
        net = self.conv5(torch.cat((inp, net1, net2, net3, net4), 1))
        return net * 0.2 + inp


class RRDB(nn.Module):
    def __init__(self, in_channels: int, grow_channels: int):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, grow_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, grow_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, grow_channels)

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.rdb1(net)
        net = self.rdb2(net)
        net = self.rdb3(net)
        return net * 0.2 + inp


def pixel_unshuffle(net: Tensor, scale: int) -> Tensor:
    b, c, h, w = net.shape
    out_channel = c * (scale**2)
    assert h % scale == 0 and w % scale == 0
    h = h // scale
    w = w // scale
    net = net.view(b, c, h, scale, w, scale)
    net = net.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, h, w)
    return net


@IDLModel.register("rrdb")
class RRDBNet(IDLModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        *,
        scale: int = 4,
        latent_channels: int = 64,
        grow_channels: int = 32,
        num_layers: int = 23,
    ):
        super().__init__()
        self.scale = scale
        out_channels = out_channels or in_channels
        if scale == 2:
            in_channels = in_channels * 4
        elif scale == 1:
            in_channels = in_channels * 16
        self.conv_first = nn.Conv2d(in_channels, latent_channels, 3, 1, 1)
        make_layer = lambda: RRDB(latent_channels, grow_channels)
        self.body = nn.Sequential(*[make_layer() for _ in range(num_layers)])
        self.conv_body = nn.Conv2d(latent_channels, latent_channels, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(latent_channels, latent_channels, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(latent_channels, latent_channels, 3, 1, 1)
        self.conv_hr = nn.Conv2d(latent_channels, latent_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(latent_channels, out_channels, 3, 1, 1)
        # misc
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = lambda net: F.interpolate(net, scale_factor=2, mode="nearest")

    def forward(self, net: Tensor) -> Tensor:
        if self.scale == 2:
            net = pixel_unshuffle(net, scale=2)
        elif self.scale == 1:
            net = pixel_unshuffle(net, scale=4)
        net = self.conv_first(net)
        body = self.conv_body(self.body(net))
        net = net + body
        # upsample
        net = self.lrelu(self.conv_up1(self.upsample(net)))
        net = self.lrelu(self.conv_up2(self.upsample(net)))
        net = self.conv_last(self.lrelu(self.conv_hr(net)))
        return net


__all__ = [
    "RRDBNet",
]
