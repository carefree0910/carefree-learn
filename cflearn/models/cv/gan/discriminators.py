import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Optional
from typing import NamedTuple

from ....protocol import WithRegister
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d


discriminator_dict: Dict[str, Type["DiscriminatorBase"]] = {}


class DiscriminatorOutput(NamedTuple):
    output: torch.Tensor
    cond_logits: Optional[torch.Tensor] = None


class DiscriminatorBase(nn.Module, WithRegister):
    d: Dict[str, Type["DiscriminatorBase"]] = discriminator_dict

    clf: nn.Module
    net: nn.Module
    cond: Optional[nn.Module]

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

    def generate_cond(self, out_channels: int) -> None:
        if self.num_classes is None:
            self.cond = None
        else:
            self.cond = Conv2d(
                out_channels,
                self.num_classes,
                kernel_size=4,
                padding=1,
                stride=1,
            )

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        feature_map = self.net(x)
        logits = self.clf(feature_map)
        cond_logits = None
        if self.cond is not None:
            cond_logits_map = self.cond(feature_map)
            cond_logits = torch.mean(cond_logits_map, [2, 3])
        return DiscriminatorOutput(logits, cond_logits)


@DiscriminatorBase.register("basic")
class NLayerDiscriminator(DiscriminatorBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: Optional[int] = None,
        *,
        num_layers: int = 2,
        start_channels: int = 16,
        norm_type: str = "batch",
    ):
        super().__init__(img_size, in_channels, num_classes)
        self.img_size = img_size
        self.num_layers = num_layers
        self.start_channels = start_channels
        # backbone
        blocks: List[nn.Module] = [
            Conv2d(
                in_channels,
                self.start_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nc_multiplier = 1
        for i in range(1, num_layers):
            nc_multiplier_prev = nc_multiplier
            nc_multiplier = min(2 ** i, 8)
            blocks.extend(
                get_conv_blocks(
                    start_channels * nc_multiplier_prev,
                    start_channels * nc_multiplier,
                    4,
                    2,
                    bias=False,
                    padding=1,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.net = nn.Sequential(*blocks)
        # heads
        out_channels = start_channels * nc_multiplier
        self.clf = Conv2d(
            out_channels,
            1,
            kernel_size=4,
            padding=1,
            stride=1,
            bias=False,
        )
        self.generate_cond(out_channels)


__all__ = [
    "DiscriminatorOutput",
    "DiscriminatorBase",
    "NLayerDiscriminator",
]
