import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import List
from typing import Optional

from ..common import register_discriminator
from ..common import IDiscriminator
from ..common import DiscriminatorOutput
from ...core import get_conv_blocks
from ...core import Conv2d


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@register_discriminator("basic")
class NLayerDiscriminator(IDiscriminator):
    def __init__(
        self,
        in_channels: int,
        num_classes: Optional[int] = None,
        *,
        num_layers: int = 2,
        start_channels: int = 16,
        norm_type: Optional[str] = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
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
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        use_bias = norm_type != "batch"
        nc_multiplier = 1
        for i in range(1, num_layers):
            nc_multiplier_prev = nc_multiplier
            nc_multiplier = min(2**i, 8)
            blocks.extend(
                get_conv_blocks(
                    start_channels * nc_multiplier_prev,
                    start_channels * nc_multiplier,
                    4,
                    1 if i == num_layers - 1 else 2,
                    bias=use_bias,
                    padding=1,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.feature_net = nn.Sequential(*blocks)
        # heads
        out_channels = start_channels * nc_multiplier
        self.head = Conv2d(
            out_channels,
            1,
            kernel_size=4,
            padding=1,
            stride=1,
            bias=True,
        )
        # conditional
        self.generate_cond(out_channels)
        # initialize
        self.apply(weights_init)

    def forward(self, net: Tensor) -> DiscriminatorOutput:
        feature_map = self.feature_net(net)
        logits = self.head(feature_map)
        cond_logits = None
        if self.cond is not None:
            cond_logits_map = self.cond(feature_map)
            cond_logits = torch.mean(cond_logits_map, [2, 3])
        return DiscriminatorOutput(logits, cond_logits)


@register_discriminator("multi_scale")
class MultiScaleDiscriminator(IDiscriminator):
    def __init__(
        self,
        in_channels: int,
        num_classes: Optional[int] = None,
        *,
        num_scales: int = 3,
        num_layers: int = 4,
        latent_channels: int = 64,
        activation: str = "leaky_relu_0.2",
        norm_type: Optional[str] = None,
    ):
        if num_classes is not None:
            msg = "`MultiScaleDiscriminator` does not support conditional inputs"
            raise ValueError(msg)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.latent_channels = latent_channels
        self.activation = activation
        self.norm_type = norm_type
        self.downsample = nn.AvgPool2d(
            3,
            stride=2,
            padding=[1, 1],
            count_include_pad=False,
        )
        nets = [self._make() for _ in range(num_scales)]
        self.discriminators = nn.ModuleList(nets)
        self.apply(weights_init)

    def forward(self, net: Tensor) -> List[DiscriminatorOutput]:
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(DiscriminatorOutput(discriminator(net)))
            net = self.downsample(net)
        return outputs

    # internal

    def _make(self) -> nn.Sequential:
        in_nc = self.in_channels
        out_nc = self.latent_channels
        blocks = []
        for i in range(self.num_layers):
            blocks.extend(
                get_conv_blocks(
                    in_nc,
                    out_nc,
                    4,
                    2,
                    norm_type=None if i == 0 else self.norm_type,
                    activation=self.activation,
                    padding="reflection1",
                )
            )
            in_nc = out_nc
            out_nc *= 2
        blocks.append(Conv2d(in_nc, 1, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*blocks)


__all__ = [
    "NLayerDiscriminator",
    "MultiScaleDiscriminator",
]
