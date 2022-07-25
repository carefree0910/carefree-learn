import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional

from ...protocols.cv import ImageTranslatorMixin
from ....modules.blocks import NormFactory
from ....misc.internal_.register import register_module


class UnetSkipConnectionBlock(nn.Module):
    """
    Dimension flows:
           input                   (input_channels)
        -> down_conv
        -> inner_net               (inner_channels)
        -> submodule
        -> sub_out                 (submodule.outer_channels)
        -> cat(inner_net, sub_out) (submodule.input_channels + submodule.outer_channels)
        -> up_conv
        -> output                  (outer_channels)

    > This means that:
    1) inner_channels should be submodule.input_channels
    2) we should restrict submodule.input_channels & submodule.outer_channels
    -> e.g. let submodule.input_channels = submodule.outer_channels = inner_channels
    """

    def __init__(
        self,
        inner_channels: int,
        outer_channels: int,
        input_channels: Optional[int] = None,
        submodule: Optional["UnetSkipConnectionBlock"] = None,
        is_outermost: bool = False,
        is_innermost: bool = False,
        norm_type: Optional[str] = "batch",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.inner_channels = inner_channels
        self.outer_channels = outer_channels
        self.input_channels = input_channels or outer_channels
        self.is_outermost = is_outermost
        msg = None
        if submodule:
            if submodule.is_outermost:
                msg = "submodule is already outermost"
            elif inner_channels != submodule.input_channels:
                msg = "`inner_channels` should equal to `input_channels` of submodule"
            elif inner_channels != submodule.outer_channels:
                msg = "`inner_channels` should equal to `outer_channels` of submodule"
        if msg is not None:
            raise ValueError(msg)
        use_bias = norm_type == "instance"
        norm_kwargs = norm_kwargs or {}
        downsample = nn.Conv2d(
            self.input_channels,
            inner_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=use_bias,
        )
        norm_factory = NormFactory(norm_type)
        if is_outermost:
            blocks = [
                downsample,
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels * 2,
                    outer_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            ]
        elif is_innermost:
            blocks = [
                nn.LeakyReLU(0.2, True),
                downsample,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels,
                    outer_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_factory.make(outer_channels, **norm_kwargs),
            ]
        else:
            blocks = [
                nn.LeakyReLU(0.2, True),
                downsample,
                norm_factory.make(inner_channels, **norm_kwargs),
                submodule,
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    inner_channels * 2,
                    outer_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_factory.make(outer_channels, **norm_kwargs),
            ]
            if use_dropout:
                blocks.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*blocks)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if self.is_outermost:
            return self.model(net)
        return torch.cat([net, self.model(net)], 1)


@register_module("unet_generator", pre_bases=[ImageTranslatorMixin])
class UnetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: int = 8,
        *,
        start_channels: int = 64,
        norm_type: Optional[str] = "batch",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_downsample = num_downsample
        self.start_channels = start_channels

        unet_block = UnetSkipConnectionBlock(
            start_channels * 8,
            start_channels * 8,
            input_channels=None,
            submodule=None,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            is_innermost=True,
        )
        for i in range(num_downsample - 5):
            unet_block = UnetSkipConnectionBlock(
                start_channels * 8,
                start_channels * 8,
                input_channels=None,
                submodule=unet_block,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            start_channels * 8,
            start_channels * 4,
            input_channels=None,
            submodule=unet_block,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )
        unet_block = UnetSkipConnectionBlock(
            start_channels * 4,
            start_channels * 2,
            input_channels=None,
            submodule=unet_block,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )
        unet_block = UnetSkipConnectionBlock(
            start_channels * 2,
            start_channels,
            input_channels=None,
            submodule=unet_block,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )
        self.model = UnetSkipConnectionBlock(
            start_channels,
            self.out_channels,
            input_channels=in_channels,
            submodule=unet_block,
            is_outermost=True,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.model(net)


__all__ = ["UnetGenerator"]
