import math

import torch.nn as nn

from typing import Any
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict

from .protocol import DecoderBase
from .protocol import Decoder1DBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Linear
from ....modules.blocks import ResidualBlock
from ....modules.blocks import ChannelPadding
from ....modules.blocks import UpsampleConv2d


@DecoderBase.register("vanilla")
class VanillaDecoder(DecoderBase):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        norm_type: Optional[str] = "instance",
        res_norm_type: Optional[str] = "instance",
        activation: Optional[str] = "leaky_relu_0.2",
        padding: str = "reflection",
        *,
        kernel_size: int = 3,
        last_kernel_size: Optional[int] = None,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        num_repeats: Union[str, List[int]] = "default",
        reduce_channel_on_upsample: bool = False,
        img_size: Optional[int] = None,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
    ):
        super().__init__(
            latent_channels,
            out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            cond_channels=cond_channels,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
        )

        blocks: List[nn.Module] = []

        for _ in range(num_residual_blocks):
            blocks.append(
                ResidualBlock(
                    latent_channels,
                    residual_dropout,
                    norm_type=res_norm_type,
                    activation=activation,
                    padding=padding,
                )
            )

        if isinstance(num_repeats, str):
            if num_repeats == "default":
                num_repeats = [0] + [1] * self.num_upsample
            elif num_repeats == "repeated":
                repeats1 = (self.num_upsample - 1) // 2
                repeats0 = self.num_upsample - repeats1 - 1
                num_repeats = [1] + [4] * repeats0 + [2] * repeats1 + [1]
            else:
                raise ValueError(f"unrecognized `num_repeats` '{num_repeats}` occurred")
        if len(num_repeats) != self.num_upsample + 1:
            msg = "length of `num_repeats` is not identical with `num_upsample + 1`"
            raise ValueError(msg)

        in_nc = latent_channels
        if self.is_conditional:
            in_nc += cond_channels

        if last_kernel_size is None:
            last_kernel_size = kernel_size
        for i, num_repeat in enumerate(num_repeats):
            is_last = i == self.num_upsample
            if is_last:
                num_repeat += 1
            if num_repeat == 0:
                continue
            repeat_channels = latent_channels if i == 0 else in_nc
            out_nc = repeat_channels // 2
            if i != 0:
                num_repeat -= 1
                if reduce_channel_on_upsample:
                    first_out_nc = out_nc
                else:
                    first_out_nc = in_nc if num_repeat != 0 else out_nc
                if is_last and num_repeat == 0:
                    first_out_nc = out_channels
                    kernel_size = last_kernel_size
                    norm_type = None
                    activation = None
                blocks.extend(
                    get_conv_blocks(
                        in_nc,
                        first_out_nc,
                        kernel_size,
                        1,
                        bias=True,
                        factor=2,
                        norm_type=norm_type,
                        activation=activation,
                        conv_base=UpsampleConv2d,
                        padding=padding,
                    )
                )
                in_nc = first_out_nc
            for j in range(num_repeat):
                if is_last and j == num_repeat - 1:
                    out_nc = out_channels
                    kernel_size = last_kernel_size
                    norm_type = None
                    activation = None
                blocks.extend(
                    get_conv_blocks(
                        in_nc,
                        repeat_channels if j != num_repeat - 1 else out_nc,
                        kernel_size,
                        1,
                        bias=True,
                        norm_type=norm_type,
                        activation=activation,
                        padding=padding,
                    )
                )
                in_nc = repeat_channels
            in_nc = out_nc

        self.decoder = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = self._inject_cond(batch)
        net = self.decoder(batch[INPUT_KEY])
        net = self.resize(net)
        return {PREDICTIONS_KEY: net}


@Decoder1DBase.register("vanilla")
class VanillaDecoder1D(Decoder1DBase):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        norm_type: Optional[str] = "instance",
        res_norm_type: Optional[str] = "instance",
        activation: Optional[str] = "leaky_relu_0.2",
        padding: str = "reflection",
        *,
        kernel_size: int = 3,
        last_kernel_size: Optional[int] = None,
        num_residual_blocks: int = 0,
        residual_dropout: float = 0.0,
        num_repeats: Union[str, List[int]] = "default",
        reduce_channel_on_upsample: bool = False,
        img_size: Optional[int] = None,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        latent_padding_channels: Optional[int] = 16,
    ):
        super().__init__(
            latent_dim,
            out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
        )
        if latent_resolution is None:
            latent_resolution = int(round(img_size / 2 ** self.num_upsample))
        self.latent_resolution = latent_resolution
        latent_area = self.latent_resolution ** 2
        latent_channels = math.ceil(latent_dim / latent_area)
        shape = -1, latent_channels, latent_resolution, latent_resolution
        blocks: List[nn.Module] = [
            Linear(latent_dim, latent_channels * latent_area),
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(latent_channels, latent_dim, kernel_size=1, bias=False),
        ]
        if latent_padding_channels is not None:
            latent_dim += latent_padding_channels
            latent_padding = ChannelPadding(latent_padding_channels, latent_resolution)
            blocks.append(latent_padding)
        self.from_latent = nn.Sequential(*blocks)
        self.decoder = VanillaDecoder(
            latent_dim,
            out_channels,
            norm_type,
            res_norm_type,
            activation,
            padding,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            num_residual_blocks=num_residual_blocks,
            residual_dropout=residual_dropout,
            num_repeats=num_repeats,
            reduce_channel_on_upsample=reduce_channel_on_upsample,
            img_size=img_size,
            num_upsample=self.num_upsample,
            cond_channels=cond_channels,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        net = batch[INPUT_KEY]
        net = self.from_latent(net)
        batch[INPUT_KEY] = net
        return self.decoder(batch_idx, batch, state, **kwargs)


__all__ = [
    "VanillaDecoder",
    "VanillaDecoder1D",
]
