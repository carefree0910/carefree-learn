import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .schema import Decoder1DMixin
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import Activation
from ....modules.blocks import UpsampleConv2d


def setup_filter(
    f: Any,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
    flip_filter: bool = False,
    gain: float = 1.0,
    separable: Optional[bool] = None,
) -> Tensor:
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


def resample(
    net: Tensor,
    f: Tensor,
    upscale: float,
    padding: List[int],
    gain: float,
) -> Tensor:
    c, h, w = map(int, net.shape[1:])
    upscale = int(upscale)
    net = net.reshape([-1, c, h, 1, w, 1])
    net = F.pad(net, [0, upscale - 1, 0, 0, 0, upscale - 1])
    net = net.reshape([-1, c, h * upscale, w * upscale])
    net = F.pad(net, padding)
    f = f * (gain ** (f.ndim / 2))
    f = f[None, None, ...]
    f = f.repeat([c, 1] + [1] * (f.ndim - 2))
    net = F.conv2d(net, f, groups=c)
    return net


class FullyConnected(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        lr_multiplier: float = 1.0,
        bias_init: float = 0.0,
        activation: Optional[str] = None,
        activation_gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([out_dim, in_dim]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_dim], bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_dim)
        self.bias_gain = lr_multiplier
        if activation is None:
            self.activation = None
        else:
            self.activation = Activation.make(activation)
        self.activation_gain = activation_gain

    def forward(self, net: Tensor) -> Tensor:
        w = self.weight * self.weight_gain
        b = self.bias
        if b is None:
            net = net.matmul(w.t())
        if b is not None:
            if self.bias_gain != 1:
                b = b * self.bias_gain
            net = torch.addmm(b.unsqueeze(0), net, w.t())
        if self.activation is not None:
            net = self.activation(net)
            net = net * self.activation_gain
        return net


class ToRGB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        kernel_size: int = 1,
        conv_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnected(w_dim, in_channels, bias_init=1.0)
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros([1, out_channels, 1, 1]))
        self.weight_gain = 1.0 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, net: Tensor, w: Tensor) -> Tensor:
        styles = self.affine(w) * self.weight_gain
        net = self.conv(net, styles) + self.bias
        if self.conv_clamp is not None:
            net = torch.clamp(net, -self.conv_clamp, self.conv_clamp)
        return net


class StyleGAN2Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        resolution: int,
        kernel_size: int = 3,
        upscale_factor: float = 1.0,
        use_noise: bool = True,
        activation: str = "leaky_relu_0.2",
        activation_gain: float = math.sqrt(2.0),
        resample_filter: Optional[List[int]] = None,
        conv_clamp: Optional[float] = None,
    ):
        super().__init__()
        if resample_filter is None:
            resample_filter = [1, 3, 3, 1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.upscale_factor = upscale_factor
        self.use_noise = use_noise
        self.activation = Activation.make(activation)
        self.activation_gain = activation_gain
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.affine = FullyConnected(latent_dim, in_channels, bias_init=1.0)
        self.conv = UpsampleConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            mode="transpose",
            bias=False,
            demodulate=True,
            factor=upscale_factor,
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros([1, out_channels, 1, 1]))

    def forward(
        self,
        net: Tensor,
        w: Tensor,
        noise_mode: str = "random",
        gain: float = 1.0,
    ) -> Tensor:
        styles = self.affine(w)
        net = self.conv(net, styles)
        if self.upscale_factor > 1.0:
            net = resample(
                net,
                self.resample_filter,
                1.0,
                [1] * 4,
                self.upscale_factor**2,
            )
        if self.use_noise:
            if noise_mode == "const":
                noise = self.noise_const
            elif noise_mode == "random":
                noise_shape = net.shape[0], 1, self.resolution, self.resolution
                noise = torch.randn(noise_shape, device=net.device)
            else:
                raise NotImplementedError(f"noise_mode '{noise_mode}' not implemented")
            noise = noise * self.noise_strength
            net.add_(noise)
        net = net + self.bias
        activation_gain = self.activation_gain * gain
        if activation_gain != 1.0:
            net = net * activation_gain
        net = self.activation(net)
        if self.conv_clamp is not None:
            conv_clamp = self.conv_clamp * gain
            net = torch.clamp(net, -conv_clamp, conv_clamp)
        return net


class StyleGAN2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        resolution: int,
        img_channels: int,
        is_last: bool,
        architecture: str = "skip",
        resample_filter: Optional[List[int]] = None,
        conv_clamp: Optional[float] = None,
        **conv_kwargs: Any,
    ):
        super().__init__()
        if resample_filter is None:
            resample_filter = [1, 3, 3, 1]
        self.in_channels = in_channels
        self.w_dim = latent_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.num_conv = 0
        self.num_to_rgb = 0

        if in_channels == 0:
            self.conv0 = None
        else:
            self.conv0 = StyleGAN2Conv(
                in_channels,
                out_channels,
                latent_dim=latent_dim,
                resolution=resolution,
                upscale_factor=2.0,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                **conv_kwargs,
            )
            self.num_conv += 1

        self.conv1 = StyleGAN2Conv(
            out_channels,
            out_channels,
            latent_dim=latent_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            **conv_kwargs,
        )
        self.num_conv += 1

        if not (is_last or architecture == "skip"):
            self.to_rgb = None
        else:
            self.to_rgb = ToRGB(
                out_channels,
                img_channels,
                w_dim=latent_dim,
                conv_clamp=conv_clamp,
            )
            self.num_to_rgb += 1

    def forward(
        self,
        net: Optional[Tensor],
        rgb: Optional[Tensor],
        ws: Tensor,
        *,
        determinate: bool = False,
        **conv_kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        w_iter = iter(ws.unbind(dim=1))
        if determinate:
            conv_kwargs.setdefault("noise_mode", "const")
        if self.conv0 is not None:
            net = self.conv0(net, next(w_iter), **conv_kwargs)
        net = self.conv1(net, next(w_iter), **conv_kwargs)
        if rgb is not None:
            rgb = resample(rgb, self.resample_filter, 2.0, [2, 1, 2, 1], 4.0)
        if self.to_rgb is not None:
            y = self.to_rgb(net, next(w_iter))
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            rgb = rgb.add_(y) if rgb is not None else y
        return net, rgb


class StyleGAN2Decoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 512,
        out_channels: int = 3,
        *,
        channel_base: int = 32768,
        max_channels: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        **block_kwargs: Any,
    ):
        assert img_size >= 4 and img_size & (img_size - 1) == 0
        super().__init__()
        self.w_dim = latent_dim
        self.img_size = img_size
        self.img_size_log2 = int(round(math.log2(img_size)))
        self.img_channels = out_channels
        self.block_resolutions = [2**i for i in range(2, self.img_size_log2 + 1)]
        channels_dict = {
            resolution: min(channel_base // resolution, max_channels)
            for resolution in self.block_resolutions
        }
        self.out_channels = out_channels
        self.num_upsample = len(self.block_resolutions)
        self.latent_resolution = self.block_resolutions[0]
        self.latent_channels = channels_dict[self.latent_resolution]
        self.num_classes = num_classes
        shape = (1 if num_classes is None else num_classes), self.latent_channels, 4, 4
        self.const = nn.Parameter(torch.randn(*shape))
        blocks = {}
        self.num_ws = 0
        for resolution in self.block_resolutions:
            in_nc = channels_dict[resolution // 2] if resolution > 4 else 0
            out_nc = channels_dict[resolution]
            is_last = resolution == self.img_size
            block = StyleGAN2Block(
                in_nc,
                out_nc,
                latent_dim=latent_dim,
                resolution=resolution,
                img_channels=out_channels,
                is_last=is_last,
                conv_clamp=conv_clamp,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_to_rgb
            blocks[str(resolution)] = block
        self.blocks = nn.ModuleDict(blocks)

    def forward(self, ws: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        w_idx = 0
        block_ws = []
        for resolution in self.block_resolutions:
            block = self.blocks[str(resolution)]
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_to_rgb))
            w_idx += block.num_conv
        rgb = None
        batch_size = ws.shape[0]
        if self.num_classes is None:
            net = self.const.repeat([batch_size, 1, 1, 1])
        else:
            if labels is None:
                labels = torch.randint(self.num_classes, [batch_size], device=ws.device)
            net = self.const[labels.view(-1)]
        for resolution, resolution_ws in zip(self.block_resolutions, block_ws):
            block = self.blocks[str(resolution)]
            net, rgb = block(net, rgb, resolution_ws, **kwargs)
        return rgb


@Decoder1DMixin.register("style2")
class StyleDecoder2(nn.Module, Decoder1DMixin):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 512,
        out_channels: int = 3,
        *,
        channel_base: int = 32768,
        max_channels: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        **block_kwargs: Any,
    ):
        assert img_size >= 4 and img_size & (img_size - 1) == 0
        super().__init__()
        self.latent_dim = latent_dim
        num_upsample = int(round(math.log2(img_size))) - 1
        self._initialize(
            out_channels=out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            num_classes=num_classes,
            latent_resolution=4,
        )
        block_kwargs.pop("num_upsample", None)
        block_kwargs.pop("latent_resolution", None)
        self.decoder = StyleGAN2Decoder(
            img_size,
            latent_dim,
            out_channels,
            channel_base=channel_base,
            max_channels=max_channels,
            num_classes=num_classes,
            conv_clamp=conv_clamp,
            **block_kwargs,
        )

    def _z2ws(self, z: Tensor) -> Tensor:
        return z.unsqueeze(1).repeat(1, self.decoder.num_ws, 1)

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        ws = self._z2ws(batch[INPUT_KEY])
        net = self.decoder(ws, labels=batch.get(LABEL_KEY), **kwargs)
        return net


__all__ = [
    "FullyConnected",
    "StyleGAN2Decoder",
    "StyleDecoder2",
]
