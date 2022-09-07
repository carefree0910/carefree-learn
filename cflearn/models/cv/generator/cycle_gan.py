from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..decoder.vanilla import VanillaDecoder
from ..encoder.vanilla import VanillaEncoder
from ...protocols.cv import ImageTranslatorMixin
from ....constants import INPUT_KEY
from ....misc.toolkit import interpolate
from ....misc.internal_.register import register_module


@register_module("cycle_gan_generator")
class CycleGANGenerator(nn.Module, ImageTranslatorMixin):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: int = 2,
        *,
        kernel_size: int = 3,
        first_kernel_size: int = 7,
        start_channels: int = 64,
        num_residual_blocks: int = 9,
        residual_dropout: float = 0.0,
        norm_type: Optional[str] = "instance",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        activation: str = "relu",
        padding: str = "reflection",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_downsample = num_downsample
        self.start_channels = start_channels
        self.latent_channels = start_channels * 2**num_downsample
        self.encoder = VanillaEncoder(
            in_channels,
            num_downsample,
            self.latent_channels,
            kernel_size=kernel_size,
            first_kernel_size=first_kernel_size,
            start_channels=self.start_channels,
            num_residual_blocks=num_residual_blocks,
            residual_dropout=residual_dropout,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            activation=activation,
            padding=padding,
        )
        self.decoder = VanillaDecoder(
            self.latent_channels,
            self.out_channels,
            norm_type=norm_type,
            activation=activation,
            padding=padding,
            kernel_size=kernel_size,
            last_kernel_size=first_kernel_size,
            num_repeats="default",
            reduce_channel_on_upsample=True,
            num_upsample=num_downsample,
        )

    def forward(self, inp: Tensor) -> Tensor:
        net = self.encoder(inp)
        net = self.decoder({INPUT_KEY: net})
        net = interpolate(net, anchor=inp)
        return net


__all__ = ["CycleGANGenerator"]
