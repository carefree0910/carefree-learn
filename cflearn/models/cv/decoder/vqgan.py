import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from .schema import DecoderMixin
from ....constants import INPUT_KEY
from ..encoder.vqgan import normalize
from ..encoder.vqgan import AttnBlock
from ..encoder.vqgan import ResidualBlock
from ....misc.toolkit import interpolate
from ....modules.blocks import Conv2d


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, net: Tensor) -> Tensor:
        net = interpolate(net, factor=2.0, mode="nearest")
        net = self.net(net)
        return net


@DecoderMixin.register("vqgan")
class VQGANDecoder(nn.Module, DecoderMixin):
    def __init__(
        self,
        img_size: int,
        out_channels: int = 3,
        latent_channels: int = 256,
        *,
        channel_multipliers: Optional[List[int]] = None,
        attention_resolutions: Optional[List[int]] = None,
        latent_dim: int = 128,
        res_dropout: float = 0.0,
        num_res_blocks: int = 3,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        if channel_multipliers is None:
            channel_multipliers = [1, 1, 2, 2, 4]
        if attention_resolutions is None:
            attention_resolutions = [16]
        if num_upsample is not None and num_upsample != len(channel_multipliers):
            raise ValueError(
                f"`num_upsample` ({num_upsample}) should be identical with "
                f"the length of `channel_multipliers` ({len(channel_multipliers)}"
            )
        num_upsample = len(channel_multipliers)
        self._initialize(
            out_channels=out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
        )
        self._init_cond(cond_channels=cond_channels)
        in_nc = latent_dim * channel_multipliers[-1]
        current_res = img_size // 2 ** (num_upsample - 1)
        # in conv
        self.in_conv = Conv2d(latent_channels, in_nc, kernel_size=3, padding=1)
        # mid
        self.mid = nn.Sequential(
            ResidualBlock(in_nc, in_nc, res_dropout),
            AttnBlock(in_nc),
            ResidualBlock(in_nc, in_nc, res_dropout),
        )
        # upsample
        blocks = []
        for i in reversed(range(num_upsample)):
            i_blocks = []
            out_nc = latent_dim * channel_multipliers[i]
            for j in range(num_res_blocks):
                i_blocks.append(ResidualBlock(in_nc, out_nc, res_dropout))
                in_nc = out_nc
                if current_res in attention_resolutions:
                    i_blocks.append(AttnBlock(in_nc))
            if i != 0:
                i_blocks.append(Upsample(in_nc))
                current_res *= 2
            blocks.append(nn.Sequential(*i_blocks))
        self.upsample_blocks = nn.ModuleList(blocks[::-1])
        # finalize
        self.out = nn.Sequential(
            normalize(in_nc),
            nn.SiLU(),
            Conv2d(in_nc, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        net = batch[INPUT_KEY]
        net = self.in_conv(net)
        net = self.mid(net)
        for block in self.upsample_blocks[::-1]:
            net = block(net)
        net = self.out(net)
        if kwargs.get("resize", True):
            net = self.resize(net)
        return net


__all__ = [
    "VQGANDecoder",
]
