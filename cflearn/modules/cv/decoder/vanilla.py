from torch import nn
from torch import Tensor
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import print_warning

from ..common import IDecoder
from ..common import DecoderInputs
from ..common import register_decoder
from ...core import get_conv_blocks
from ...core import build_activation
from ...core import Linear
from ...core import ResidualBlock
from ...core import ChannelPadding
from ...core import UpsampleConv2d
from ...common import Lambda
from ....toolkit import auto_num_layers


@register_decoder("vanilla")
class VanillaDecoder(IDecoder):
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
        apply_tanh: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        if num_upsample is None:
            msg_fmt = "`{}` should be provided when `num_upsample` is not"
            if img_size is None:
                raise ValueError(msg_fmt.format("img_size"))
            if latent_resolution is None:
                print_warning(
                    f'{msg_fmt.format("latent_resolution")}, '
                    "and 7 will be used as the default `latent_resolution` now"
                )
                latent_resolution = 7
            num_upsample = auto_num_layers(img_size, latent_resolution, None)
        self.num_upsample = num_upsample
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_resolution = latent_resolution
        self.generate_cond(cond_channels=cond_channels)

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

        if last_kernel_size is None:
            last_kernel_size = kernel_size
        in_nc = latent_channels
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

        self.decoder = nn.ModuleList(blocks)
        self.apply_tanh = apply_tanh

    def forward(self, inputs: DecoderInputs) -> Tensor:
        net = self.inject_cond(inputs.z, inputs.labels)
        for block in self.decoder:
            if not isinstance(block, UpsampleConv2d):
                net = block(net)
            else:
                net = block(net, deterministic=inputs.deterministic)
        net = self.resize(net, deterministic=inputs.deterministic)
        return net


@register_decoder("vanilla_1d")
class VanillaDecoder1D(IDecoder):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        latent_channels: int = 64,
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
        latent_padding_channels: Optional[int] = None,
        apply_tanh: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        if num_upsample is None:
            fmt = "`{}` should be provided when `num_upsample` is not"
            if img_size is None:
                raise ValueError(fmt.format("img_size"))
            if latent_resolution is None:
                print_warning(
                    f'{fmt.format("latent_resolution")}, '
                    "and 7 will be used as the default `latent_resolution` now"
                )
                latent_resolution = 7
            num_upsample = auto_num_layers(img_size, latent_resolution, None)
        self.num_upsample = num_upsample
        self.img_size = img_size
        self.num_classes = num_classes
        if latent_resolution is None:
            latent_resolution = int(round(img_size / 2**self.num_upsample))
        self.latent_resolution = latent_resolution
        assert isinstance(self.latent_resolution, int)
        latent_area = self.latent_resolution**2
        shape = -1, latent_channels, latent_resolution, latent_resolution
        blocks: List[nn.Module] = [
            Linear(latent_dim, latent_channels * latent_area),
            build_activation(activation),
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
        ]
        if latent_padding_channels is not None:
            latent_padding = ChannelPadding(
                latent_channels,
                latent_padding_channels,
                latent_resolution,
            )
            blocks.append(latent_padding)
        self.from_latent = nn.Sequential(*blocks)
        self.decoder = VanillaDecoder(
            latent_channels,
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
            apply_tanh=apply_tanh,
        )

    def forward(self, inputs: DecoderInputs) -> Tensor:
        inputs.z = self.from_latent(inputs.z)
        return self.decoder.decode(inputs)


__all__ = [
    "VanillaDecoder",
    "VanillaDecoder1D",
]
