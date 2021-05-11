import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional

from .protocol import EncoderBase
from .protocol import Encoder1DBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import _get_clones
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import ResidualBlock


@EncoderBase.register("vanilla")
class VanillaEncoder(EncoderBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_downsample: int,
        latent_channels: int = 128,
        first_kernel_size: int = 7,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        norm_type: str = "instance",
    ):
        super().__init__(img_size, in_channels, num_downsample, latent_channels)
        # downsample
        self.first_kernel_size = first_kernel_size
        start_channels = int(round(latent_channels / (2 ** self.num_downsample)))
        if start_channels <= 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) is too small "
                f"for num_downsample ({self.num_downsample})"
            )
        blocks = get_conv_blocks(
            in_channels,
            start_channels,
            first_kernel_size,
            1,
            bias=False,
            norm_type=norm_type,
            activation=nn.LeakyReLU(0.2, inplace=True),
        )
        in_nc = start_channels
        for i in range(self.num_downsample):
            is_last = i == self.num_downsample - 1
            if is_last:
                out_nc = latent_channels
            else:
                out_nc = min(in_nc * 2, latent_channels)
            if is_last:
                blocks.append(
                    Conv2d(in_nc, out_nc, kernel_size=3, stride=2, bias=False)
                )
            else:
                blocks.extend(
                    get_conv_blocks(
                        in_nc,
                        out_nc,
                        3,
                        2,
                        bias=False,
                        norm_type=norm_type,
                        activation=nn.LeakyReLU(0.2, inplace=True),
                    )
                )
            in_nc = out_nc
        # residual
        if num_residual_blocks > 0:
            blocks.extend(
                _get_clones(
                    ResidualBlock(
                        in_nc,
                        residual_dropout,
                        **(residual_kwargs or {}),
                    ),
                    num_residual_blocks,
                    return_list=True,
                )
            )
        # construct
        self.encoder = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.encoder(batch[INPUT_KEY])}


@Encoder1DBase.register("vanilla")
class VanillaEncoder1D(Encoder1DBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_downsample: int,
        latent_dim: int = 128,
        first_kernel_size: int = 7,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        norm_type: str = "batch",
    ):
        super().__init__(img_size, in_channels, latent_dim)
        self.encoder = VanillaEncoder(
            img_size,
            in_channels,
            num_downsample,
            latent_dim,
            first_kernel_size,
            num_residual_blocks,
            residual_dropout,
            residual_kwargs,
            norm_type,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        encoding = self.encoder(batch_idx, batch, state, **kwargs)
        net = encoding[PREDICTIONS_KEY]
        net = self.pool(net).squeeze()
        return {PREDICTIONS_KEY: net}


__all__ = [
    "VanillaEncoder",
    "VanillaEncoder1D",
]
