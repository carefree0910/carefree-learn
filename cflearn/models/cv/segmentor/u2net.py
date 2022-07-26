import torch

import torch.nn as nn

from torch import Tensor
from typing import List
from typing import Union
from typing import Optional

from ...protocols.cv import ImageTranslatorMixin
from ....misc.toolkit import interpolate
from ....modules.blocks import get_clones
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import Interpolate
from ....misc.internal_.register import register_module


class ConvSeq(nn.Sequential):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, dilation: int = 1):
        super().__init__(
            *get_conv_blocks(
                in_channels,
                out_channels,
                3,
                1,
                norm_type="batch",
                activation=nn.ReLU(inplace=True),
                padding=dilation,
                dilation=dilation,
            )
        )


class UNetBase(nn.Module):
    in_block: nn.Module
    down_blocks: nn.ModuleList
    last_down: nn.Module
    up_blocks: nn.ModuleList
    return_up_nets: bool = False

    def forward(
        self,
        net: Tensor,
        *,
        determinate: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        net = in_net = self.in_block(net)
        down_nets = []
        for down_block in self.down_blocks:
            net = down_block(net)
            down_nets.append(net)
        inner = self.last_down(net)
        up_nets = [inner]
        for down_net, up_block in zip(down_nets[::-1], self.up_blocks):
            if inner.shape != down_net.shape:
                inner = interpolate(
                    inner,
                    anchor=down_net,
                    mode="bilinear",
                    determinate=determinate,
                )
            inner = torch.cat([inner, down_net], dim=1)
            if not isinstance(up_block, nn.ModuleList):
                inner = up_block(inner)
            else:
                for block in up_block:
                    kw = {}
                    if isinstance(block, Interpolate):
                        kw["determinate"] = determinate
                    inner = block(inner, **kw)
            up_nets.append(inner)
        if self.return_up_nets:
            return up_nets
        return inner + in_net


class UNetRS(UNetBase):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
        *,
        num_layers: int,
        inner_upsample_mode: str = "bilinear",
        inner_upsample_factor: Optional[int] = None,
    ):
        super().__init__()
        self.msg = f"{in_channels}, {mid_channels}, {out_channels}, {num_layers}"
        self.in_block = ConvSeq(in_channels, out_channels, dilation=1)
        blocks = [ConvSeq(out_channels, mid_channels, dilation=1)]
        for _ in range(num_layers - 2):
            blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    ConvSeq(mid_channels, mid_channels, dilation=1),
                )
            )
        self.down_blocks = nn.ModuleList(blocks)
        self.last_down = ConvSeq(mid_channels, mid_channels, dilation=2)
        basic_block = ConvSeq(mid_channels * 2, mid_channels, dilation=1)
        if inner_upsample_factor is not None:
            basic_block = nn.ModuleList(
                [
                    basic_block,
                    Interpolate(inner_upsample_factor, inner_upsample_mode),
                ]
            )
        blocks = get_clones(basic_block, num_layers - 2, return_list=True)
        blocks.append(ConvSeq(mid_channels * 2, out_channels, dilation=1))
        self.up_blocks = nn.ModuleList(blocks)

    def extra_repr(self) -> str:
        return self.msg


class UNetFRS(UNetBase):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
        *,
        num_layers: int,
    ):
        super().__init__()
        self.msg = f"{in_channels}, {mid_channels}, {out_channels}, {num_layers}"
        self.in_block = ConvSeq(in_channels, out_channels, dilation=1)
        blocks = [ConvSeq(out_channels, mid_channels, dilation=1)]
        for i in range(num_layers - 2):
            blocks.append(ConvSeq(mid_channels, mid_channels, dilation=2 ** (i + 1)))
        self.down_blocks = nn.ModuleList(blocks)
        self.last_down = ConvSeq(mid_channels, mid_channels, dilation=8)
        blocks = []
        for i in range(num_layers - 2):
            dilation = 2 ** (num_layers - i - 2)
            blocks.append(ConvSeq(mid_channels * 2, mid_channels, dilation=dilation))
        blocks.append(ConvSeq(mid_channels * 2, out_channels, dilation=1))
        self.up_blocks = nn.ModuleList(blocks)

    def extra_repr(self) -> str:
        return self.msg


@register_module("u2net")
class U2Net(UNetBase, ImageTranslatorMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        upsample_mode: str = "bilinear",
        latent_channels: int = 32,
        num_layers: int = 5,
        num_inner_layers: int = 7,
        lite: bool = False,
    ):
        super().__init__()
        self.return_up_nets = True

        mid_nc = latent_channels // 2 if lite else latent_channels
        out_nc = latent_channels * 2
        in_nc = out_nc if lite else latent_channels
        current_layers = num_inner_layers

        self.in_block = nn.Identity()
        blocks = [
            UNetRS(
                in_channels,
                mid_nc,
                out_nc,
                num_layers=current_layers,
                inner_upsample_mode=upsample_mode,
                inner_upsample_factor=2,
            ),
        ]
        for i in range(num_layers - 2):
            if lite:
                in_nc = out_nc
            else:
                in_nc = out_nc
                out_nc *= 2
            current_layers = num_inner_layers - i - 1
            blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2),
                    UNetRS(
                        in_nc,
                        mid_nc,
                        out_nc,
                        num_layers=current_layers,
                        inner_upsample_mode=upsample_mode,
                        inner_upsample_factor=2,
                    ),
                )
            )
            if not lite:
                mid_nc *= 2
        if not lite:
            in_nc *= 2
        blocks.append(
            nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers),
            )
        )
        self.down_blocks = nn.ModuleList(blocks)
        self.last_down = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers),
        )
        in_nc *= 2
        ncs = [out_nc] * 2
        blocks = [UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers)]
        for i in range(num_layers - 2):
            if not lite:
                mid_nc //= 2
                out_nc //= 2
            blocks.append(
                UNetRS(
                    in_nc,
                    mid_nc,
                    out_nc,
                    num_layers=current_layers,
                    inner_upsample_mode=upsample_mode,
                    inner_upsample_factor=2,
                )
            )
            current_layers += 1
            if not lite:
                in_nc //= 2
            ncs.append(out_nc)
        if not lite:
            mid_nc //= 2
        blocks.append(
            UNetRS(
                in_nc,
                mid_nc,
                out_nc,
                num_layers=current_layers,
                inner_upsample_mode=upsample_mode,
                inner_upsample_factor=2,
            )
        )
        ncs.append(out_nc)
        self.up_blocks = nn.ModuleList(blocks)

        blocks = []
        for i, nc in enumerate(ncs[::-1]):
            blocks.append(Conv2d(nc, out_channels, kernel_size=3, padding=1))
        self.side_blocks = nn.ModuleList(blocks)
        self.out = Conv2d((len(ncs)) * out_channels, out_channels, kernel_size=1)

    def forward(self, net: Tensor, *, determinate: bool = False) -> List[Tensor]:
        up_nets = super().forward(net)
        side_nets: List[Tensor] = []
        for up_net, side_block in zip(up_nets[::-1], self.side_blocks):
            side_net = side_block(up_net)
            if side_nets and side_net.shape != side_nets[0].shape:
                side_net = interpolate(
                    side_net,
                    anchor=side_nets[0],
                    mode="bilinear",
                    determinate=determinate,
                )
            side_nets.append(side_net)
        side_nets.insert(0, self.out(torch.cat(side_nets, dim=1)))
        return side_nets


__all__ = ["U2Net"]
