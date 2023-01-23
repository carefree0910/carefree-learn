import math
import torch

from typing import *
from torch import nn
from torch import Tensor
from torch.nn import init
from cftool.types import tensor_dict_type

from .schema import Decoder1DMixin
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import auto_num_layers
from ....modules.blocks import Conv2d
from ....modules.blocks import Linear
from ....modules.blocks import GaussianBlur3
from ....modules.blocks import Activation
from ....modules.blocks import NormFactory
from ....modules.blocks import UpsampleConv2d


def style_mod(net: Tensor, style: Tensor) -> Tensor:
    style = style.view(style.shape[0], 2, net.shape[1], 1, 1)
    s0, s1 = style[:, 0], style[:, 1]
    return s1 + net * (s0 + 1.0)


class DecodeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        has_first_conv: bool,
        demodulate: bool,
        upscale: bool,
        activation: nn.Module,
        layer: int = 0,
    ):
        super().__init__()
        self.layer = layer
        self.activation = activation
        self.conv_1 = None
        if not has_first_conv:
            if in_channels != out_channels:
                raise ValueError(
                    "`in_channels` should equal to `out_channels` "
                    "when `has_first_conv` is False"
                )
        else:
            if not upscale:
                self.conv_1 = Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    demodulate=demodulate,
                )
            else:
                self.conv_1 = UpsampleConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    transform_kernel=True,
                    padding="reflection",
                    bias=False,
                    demodulate=demodulate,
                    factor=2,
                )

        norm_factory = NormFactory("instance")
        norm_kwargs = {"affine": False, "eps": 1e-8}

        self.blur = GaussianBlur3(out_channels)
        self.noise_weight_1 = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        if demodulate:
            self.instance_norm_1 = None
        else:
            self.instance_norm_1 = norm_factory.make(out_channels, **norm_kwargs)
        self.style_1 = Linear(latent_dim, 2 * out_channels)

        self.conv_2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            demodulate=demodulate,
        )
        self.noise_weight_2 = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.empty(1, out_channels, 1, 1))
        if demodulate:
            self.instance_norm_2 = None
        else:
            self.instance_norm_2 = norm_factory.make(out_channels, **norm_kwargs)
        self.style_2 = Linear(latent_dim, 2 * out_channels)

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def noise(self, net: Tensor, weight: Tensor, use_noise: bool) -> Tensor:
        if not use_noise:
            s = math.pow(self.layer + 1, 0.5)
            s = s * torch.exp(-net * net / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
            net = net + s
        else:
            if use_noise == "batch_constant":
                shape = 1, 1, net.shape[2], net.shape[3]

            else:
                shape = net.shape[0], 1, net.shape[2], net.shape[3]
            noise_tensor = torch.randn(*shape)
            net = net + weight * noise_tensor.to(weight.device)
        return net

    def forward(
        self,
        net: Tensor,
        s1: Tensor,
        s2: Tensor,
        user_noise: bool = False,
    ) -> Tensor:
        if self.conv_1 is not None:
            net = self.conv_1(net)
            net = self.blur(net)

        net = self.noise(net, self.noise_weight_1, user_noise)
        net = net + self.bias_1
        if self.instance_norm_1 is not None:
            net = self.instance_norm_1(net)
        net = self.activation(net)
        net = style_mod(net, self.style_1(s1))
        net = self.conv_2(net)

        net = self.noise(net, self.noise_weight_2, user_noise)
        net = net + self.bias_2
        if self.instance_norm_2 is not None:
            net = self.instance_norm_2(net)
        net = self.activation(net)
        net = style_mod(net, self.style_2(s2))

        return net


class ToOutput(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.mapping = Conv2d(in_channels, out_channels, kernel_size=1, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)


@Decoder1DMixin.register("style")
class StyleDecoder(nn.Module, Decoder1DMixin):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 128,
        out_channels: int = 3,
        *,
        start_channels: int = 64,
        latent_resolution: int = 4,
        max_channels: int = 256,
        num_classes: Optional[int] = None,
        noise: bool = False,
        demodulate: bool = False,
        num_upsample: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        target_num_upsample = auto_num_layers(img_size, latent_resolution, None) + 1
        if num_upsample is not None and num_upsample != target_num_upsample:
            raise ValueError(
                f"`num_upsample` is provided ({num_upsample}), "
                f"but it should be {target_num_upsample}"
            )
        num_upsample = target_num_upsample
        self._initialize(
            out_channels=out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
        )
        self.noise = noise
        self.demodulate = demodulate
        self.start_channels = start_channels

        mul = 2 ** (num_upsample - 1)
        in_nc = min(max_channels, start_channels * mul)
        num_type = 1 if num_classes is None else num_classes
        resolution = latent_resolution
        self.const = nn.Parameter(torch.empty(num_type, in_nc, resolution, resolution))
        init.ones_(self.const.data)

        blocks: List[nn.Module] = []
        resolution //= 2
        for i in range(num_upsample):
            has_first_conv = i != 0
            upscale = resolution < img_size
            out_nc = min(max_channels, start_channels * mul)
            activation = Activation.make("leaky_relu_0.2" if upscale else "relu")
            blocks.append(
                DecodeBlock(
                    in_nc,
                    out_nc,
                    self.latent_dim,
                    has_first_conv,
                    self.demodulate,
                    upscale,
                    activation,
                    layer=i,
                )
            )
            resolution *= 2
            in_nc = out_nc
            mul //= 2

        self.blocks = nn.ModuleList(blocks)
        self.to_output = ToOutput(in_nc, out_channels, gain=0.03)

    def forward(self, batch: tensor_dict_type) -> Tensor:
        styles = batch.get(INPUT_KEY)
        if styles is None:
            msg = "`styles` should be provided in batch.data in `StyleGenerator`"
            raise ValueError(msg)

        labels = batch.get(LABEL_KEY)
        condition = batch.get(LATENT_KEY)
        if condition is not None:
            net = condition
        elif not self.is_conditional:
            net = self.const
        else:
            if labels is None:
                msg = "Conditional StyleDecoder is defined but labels are not provided"
                raise ValueError(msg)
            net = self.const[labels.view(-1)]

        for i in range(self.num_upsample):
            net = self.blocks[i](net, styles, styles, self.noise)  # type: ignore
        net = self.to_output(net)
        return net


__all__ = [
    "StyleDecoder",
]
