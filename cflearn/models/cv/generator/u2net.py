import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Union
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import align_to
from ....misc.toolkit import normalize_image
from ....modules.blocks import _get_clones
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d


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

    def forward(self, net: Tensor) -> Union[Tensor, List[Tensor]]:
        net = in_net = self.in_block(net)
        down_nets = []
        for down_block in self.down_blocks:
            net = down_block(net)
            down_nets.append(net)
        inner = self.last_down(down_nets[-1])
        up_nets = [inner]
        for down_net, up_block in zip(down_nets[::-1], self.up_blocks):
            if inner.shape != down_net.shape:
                inner = align_to(inner, anchor=down_net, mode="bilinear")
            inner = up_block(torch.cat([inner, down_net], dim=1))
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
    ):
        super().__init__()
        self.msg = f"{in_channels}, {mid_channels}, {out_channels}, {num_layers}"
        self.in_block = ConvSeq(in_channels, out_channels, dilation=1)
        blocks = [ConvSeq(out_channels, mid_channels, dilation=1)]
        for _ in range(num_layers - 2):
            blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),
                    ConvSeq(mid_channels, mid_channels, dilation=1),
                )
            )
        self.down_blocks = nn.ModuleList(blocks)
        self.last_down = ConvSeq(mid_channels, mid_channels, dilation=2)
        blocks = _get_clones(
            ConvSeq(mid_channels * 2, mid_channels, dilation=1),
            num_layers - 2,
            return_list=True,
        )
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


class U2NetCore(UNetBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        latent_channels: int = 32,
        num_layers: int = 5,
        max_layers: int = 7,
        use_tiny: bool = False,
    ):
        super().__init__()
        self.return_up_nets = True

        mid_nc = latent_channels // 2 if use_tiny else latent_channels
        out_nc = latent_channels * 2
        in_nc = out_nc if use_tiny else latent_channels
        current_layers = max_layers

        self.in_block = nn.Identity()
        blocks = [UNetRS(in_channels, mid_nc, out_nc, num_layers=current_layers)]
        for i in range(num_layers - 2):
            if use_tiny:
                in_nc = out_nc
            else:
                in_nc = out_nc
                out_nc *= 2
            current_layers = max_layers - i - 1
            blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2, stride=2, ceil_mode=True),
                    UNetRS(in_nc, mid_nc, out_nc, num_layers=current_layers),
                )
            )
            if not use_tiny:
                mid_nc *= 2
        if not use_tiny:
            in_nc *= 2
        blocks.append(
            nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers),
            )
        )
        self.down_blocks = nn.ModuleList(blocks)
        self.last_down = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers),
        )
        in_nc *= 2
        ncs = [out_nc] * 2
        blocks = [UNetFRS(in_nc, mid_nc, out_nc, num_layers=current_layers)]
        for i in range(num_layers - 2):
            if not use_tiny:
                mid_nc //= 2
                out_nc //= 2
            blocks.append(UNetRS(in_nc, mid_nc, out_nc, num_layers=current_layers))
            current_layers += 1
            if not use_tiny:
                in_nc //= 2
            ncs.append(out_nc)
        if not use_tiny:
            mid_nc //= 2
        blocks.append(UNetRS(in_nc, mid_nc, out_nc, num_layers=current_layers))
        ncs.append(out_nc)
        self.up_blocks = nn.ModuleList(blocks)

        ncs = ncs[::-1]
        blocks = [Conv2d(nc, out_channels, kernel_size=3, padding=1) for nc in ncs]
        self.side_blocks = nn.ModuleList(blocks)
        self.out = Conv2d((len(ncs)) * out_channels, out_channels, kernel_size=1)

    def forward(self, net: Tensor) -> List[Tensor]:
        up_nets = super().forward(net)
        side_nets: List[Tensor] = []
        for up_net, side_block in zip(up_nets[::-1], self.side_blocks):
            side_net = side_block(up_net)
            if side_nets:
                side_net = align_to(side_net, anchor=side_nets[0], mode="bilinear")
            side_nets.append(side_net)
        side_nets.append(self.out(torch.cat(side_nets, dim=1)))
        return list(map(torch.sigmoid, side_nets))


@ModelProtocol.register("u2net")
class U2Net(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        latent_channels: int = 32,
        num_layers: int = 5,
        max_layers: int = 7,
        use_tiny: bool = False,
    ):
        super().__init__()
        self.core = U2NetCore(
            in_channels,
            out_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            max_layers=max_layers,
            use_tiny=use_tiny,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.core(batch[INPUT_KEY])}

    def generate_from(self, net: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        predictions = self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]
        return normalize_image(predictions[0])


__all__ = ["U2Net"]
