import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

from .constants import STYLE_LABEL_KEY
from ..protocol import ImageTranslatorMixin
from ....protocol import ModelProtocol
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import Activations
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
    b, c, h, w = net.shape
    upscale = int(upscale)
    net = net.reshape([b, c, h, 1, w, 1])
    net = F.pad(net, [0, upscale - 1, 0, 0, 0, upscale - 1])
    net = net.reshape([b, c, h * upscale, w * upscale])
    net = F.pad(net, padding)
    f = f * (gain ** (f.ndim / 2))
    f = f[None, None, ...]
    f = f.repeat([c, 1] + [1] * (f.ndim - 2))
    net = F.conv2d(net, f, groups=c)
    return net


def normalize_z(net: Tensor, dim: int = 1, eps: float = 1.0e-8) -> Tensor:
    return net * (net.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


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
            self.activation = Activations.make(activation)
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


class MappingNetwork(nn.Module):
    def __init__(
        self,
        num_ws: int,
        *,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        num_layers: int = 8,
        embed_features: Optional[int] = None,
        layer_features: Optional[int] = None,
        activation: str = "leaky_relu_0.2",
        lr_multiplier: float = 0.01,
        w_avg_beta: float = 0.995,
    ):
        super().__init__()
        self.num_ws = num_ws
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = latent_dim
        if num_classes is None:
            embed_features = 0
        if layer_features is None:
            layer_features = latent_dim
        features_list = (
            [latent_dim + embed_features]
            + [layer_features] * (num_layers - 1)
            + [latent_dim]
        )

        if num_classes is None:
            self.embed = None
        else:
            self.embed = FullyConnected(num_classes, embed_features)
        blocks = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            blocks.append(
                FullyConnected(
                    in_features,
                    out_features,
                    lr_multiplier=lr_multiplier,
                    activation=activation,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.register_buffer("w_avg", torch.zeros([latent_dim]))

    def forward(
        self,
        z: Tensor,
        labels: Optional[Tensor],
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        skip_w_avg_update: bool = False,
    ) -> Tensor:
        net = normalize_z(z)
        if self.embed is not None:
            if labels is None:
                raise ValueError("`labels` should be provided for conditional mapping")
            y = normalize_z(self.embed(labels.to(torch.float32)))
            net = torch.cat([net, y], dim=1)
        net = self.net(net)
        if self.training and not skip_w_avg_update:
            updated = net.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            self.w_avg.copy_(updated)
        net = net.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1.0:
            if truncation_cutoff is None:
                net = self.w_avg.lerp(net, truncation_psi)
            else:
                new = self.w_avg.lerp(net[:, :truncation_cutoff], truncation_psi)
                net[:, :truncation_cutoff] = new
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
        self.weight_gain = 1.0 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, net: Tensor, w: Tensor) -> Tensor:
        styles = self.affine(w) * self.weight_gain
        net = self.conv(net, styles) + self.bias
        if self.conv_clamp is not None:
            net = torch.clamp(net, -self.conv_clamp, self.conv_clamp)
        return net


class StyleGANConv(nn.Module):
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
        self.activation = Activations.make(activation)
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
                self.upscale_factor ** 2,
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


class StyleGANBlock(nn.Module):
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
            self.conv0 = StyleGANConv(
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

        self.conv1 = StyleGANConv(
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
        **conv_kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        w_iter = iter(ws.unbind(dim=1))
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


class StyleGANDecoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 512,
        out_channels: int = 3,
        *,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        **block_kwargs: Any,
    ):
        assert img_size >= 4 and img_size & (img_size - 1) == 0
        super().__init__()
        self.w_dim = latent_dim
        self.img_size = img_size
        self.img_size_log2 = int(np.log2(img_size))
        self.img_channels = out_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_size_log2 + 1)]
        channels_dict = {
            resolution: min(channel_base // resolution, channel_max)
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
            block = StyleGANBlock(
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


@ModelProtocol.register("style_gan_generator")
class StyleGANGenerator(ImageTranslatorMixin, ModelProtocol):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 512,
        out_channels: int = 3,
        *,
        num_layers: int = 8,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_style_classes: Optional[int] = None,
        num_content_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        block_kwargs: Optional[Dict[str, Any]] = None,
        mapping_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_style_classes = num_style_classes
        self.num_content_classes = num_content_classes
        self.decoder = StyleGANDecoder(
            img_size,
            latent_dim,
            out_channels,
            channel_base=channel_base,
            channel_max=channel_max,
            num_classes=num_content_classes,
            conv_clamp=conv_clamp,
            **(block_kwargs or {}),
        )
        self.mapping = MappingNetwork(
            self.decoder.num_ws,
            latent_dim=latent_dim,
            num_classes=num_style_classes,
            num_layers=num_layers,
            **(mapping_kwargs or {}),
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z = batch[INPUT_KEY]
        truncation_psi = kwargs.pop("truncation_psi", 1.0)
        truncation_cutoff = kwargs.pop("truncation_cutoff", None)
        style_labels = batch.get(STYLE_LABEL_KEY)
        ws = self.mapping(z, style_labels, truncation_psi, truncation_cutoff)
        content_labels = batch.get(LABEL_KEY)
        rgb = self.decoder(ws, labels=content_labels, **kwargs)
        return {PREDICTIONS_KEY: rgb}


__all__ = [
    "StyleGANDecoder",
    "StyleGANGenerator",
]
