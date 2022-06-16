import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional
from cftool.array import squeeze

from .protocol import EncoderBase
from .protocol import Encoder1DBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Linear
from ....modules.blocks import ResidualBlock


# TODO : Try ResidualBlockV2


@EncoderBase.register("vanilla")
class VanillaEncoder(EncoderBase):
    def __init__(
        self,
        in_channels: int,
        num_downsample: int,
        latent_channels: int = 256,
        *,
        kernel_size: int = 3,
        first_kernel_size: int = 7,
        start_channels: Optional[int] = None,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        norm_type: Optional[str] = "batch",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        activation: str = "leaky_relu_0.2",
        padding: str = "same",
    ):
        super().__init__(in_channels, num_downsample, latent_channels)
        self.first_kernel_size = first_kernel_size
        if start_channels is None:
            start_channels = int(round(latent_channels / (2**self.num_downsample)))
        if start_channels <= 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) is too small "
                f"for num_downsample ({self.num_downsample})"
            )
        # in conv
        blocks = get_conv_blocks(
            in_channels,
            start_channels,
            first_kernel_size,
            1,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            activation=activation,
            padding=padding,
        )
        # downsample
        downsample_padding = padding
        if padding == "reflection":
            downsample_padding = f"reflection{(kernel_size - 1) // 2}"
        in_nc = start_channels
        for i in range(self.num_downsample):
            is_last = i == self.num_downsample - 1
            i_norm_type = norm_type
            i_activation = activation
            if not is_last:
                out_nc = min(in_nc * 2, latent_channels)
            else:
                out_nc = latent_channels
                if num_residual_blocks == 0:
                    i_norm_type = None
                    i_activation = None  # type: ignore
            blocks.extend(
                get_conv_blocks(
                    in_nc,
                    out_nc,
                    kernel_size,
                    2,
                    norm_type=i_norm_type,
                    activation=i_activation,
                    padding=downsample_padding,
                )
            )
            in_nc = out_nc
        # residual
        for _ in range(num_residual_blocks):
            blocks.append(
                ResidualBlock(
                    latent_channels,
                    residual_dropout,
                    norm_type=norm_type,
                    activation=activation,
                    padding=padding,
                    **(residual_kwargs or {}),
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
        return {LATENT_KEY: self.encoder(batch[INPUT_KEY])}


@Encoder1DBase.register("vanilla")
class VanillaEncoder1D(Encoder1DBase):
    def __init__(
        self,
        in_channels: int,
        num_downsample: int,
        latent_dim: int = 128,
        *,
        img_size: Optional[int] = None,
        kernel_size: int = 3,
        first_kernel_size: int = 7,
        start_channels: Optional[int] = None,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        norm_type: Optional[str] = "batch",
        activation: str = "leaky_relu_0.2",
        padding: str = "same",
        pool: str = "average",
    ):
        super().__init__(in_channels, latent_dim)
        self.encoder = VanillaEncoder(
            in_channels,
            num_downsample,
            latent_dim,
            kernel_size=kernel_size,
            first_kernel_size=first_kernel_size,
            start_channels=start_channels,
            num_residual_blocks=num_residual_blocks,
            residual_dropout=residual_dropout,
            residual_kwargs=residual_kwargs,
            norm_type=norm_type,
            activation=activation,
            padding=padding,
        )
        if pool == "average":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == "fc":
            if img_size is None:
                raise ValueError("`img_size` should be provided if `pool`=fc")
            res = self.encoder.latent_resolution(img_size)
            flattened_dim = latent_dim * res**2
            self.pool = nn.Sequential(nn.Flatten(1), Linear(flattened_dim, latent_dim))
        else:
            raise ValueError(f"unrecognized `pool` value : '{pool}'")

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.encoder(batch_idx, batch, state, **kwargs)[LATENT_KEY]
        net = squeeze(self.pool(net))
        return {LATENT_KEY: net}


__all__ = [
    "VanillaEncoder",
    "VanillaEncoder1D",
]
