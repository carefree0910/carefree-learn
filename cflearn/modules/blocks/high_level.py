import torch

import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Optional
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import WithRegister
from cftool.array import squeeze

from .convs import get_conv_blocks
from .convs import Conv2d
from .norms import NormFactory
from .customs import Linear
from .attentions import Attention
from ...misc.toolkit import get_device
from ...misc.toolkit import eval_context


class PreNorm(Module):
    def __init__(
        self,
        *dims: int,
        module: Module,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.norms = ModuleList([])
        for dim in dims:
            self.norms.append(NormFactory(norm_type).make(dim, **norm_kwargs))
        self.module = module

    def forward(self, *xs: Tensor, **kwargs: Any) -> Tensor:
        x_list = [norm(x) for x, norm in zip(xs, self.norms)]
        if not issubclass(self.module.__class__, Attention):
            return self.module(*x_list, **kwargs)
        if len(x_list) == 1:
            x_list = [x_list[0]] * 3
        elif len(x_list) == 2:
            x_list.append(x_list[1])
        if len(x_list) != 3:
            raise ValueError("there should be three inputs for `Attention`")
        return_attention = kwargs.pop("return_attention", False)
        attention_outputs = self.module(*x_list, **kwargs)
        if return_attention:
            return attention_outputs.weights
        return attention_outputs.output


class ChannelPadding(Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        map_dim: Optional[int] = None,
        *,
        is_1d: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.map_dim = map_dim
        self.is_global = map_dim is None
        self.is_conditional = num_classes is not None
        if self.is_global:
            map_dim = 1
        token_shape = (num_classes or 1), latent_channels, map_dim, map_dim
        self.channel_padding = nn.Parameter(torch.randn(*token_shape))  # type: ignore
        in_nc = in_channels + latent_channels
        if is_1d:
            self.mapping = Linear(in_nc, in_channels, bias=False)
        else:
            self.mapping = Conv2d(in_nc, in_channels, kernel_size=1, bias=False)

    def forward(self, net: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if not self.is_conditional:
            padding = self.channel_padding.repeat(net.shape[0], 1, 1, 1)
        else:
            if labels is None:
                msg = "`labels` should be provided in conditional `ChannelPadding`"
                raise ValueError(msg)
            padding = self.channel_padding[labels.view(-1)]
        if self.is_global:
            if len(net.shape) == 2:
                padding = squeeze(padding)
            else:
                padding = padding.repeat(1, 1, *net.shape[-2:])
        net = torch.cat([net, padding], dim=1)
        net = self.mapping(net)
        return net

    def extra_repr(self) -> str:
        dim_str = f"{self.in_channels}+{self.latent_channels}"
        map_dim_str = "global" if self.is_global else f"{self.map_dim}x{self.map_dim}"
        return f"{dim_str}, {map_dim_str}"


to_patches: Dict[str, Type["ImgToPatches"]] = {}


class ImgToPatches(Module, WithRegister["ImgToPatches"]):
    d = to_patches

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        """should return patches and its hw"""

    @property
    def num_patches(self) -> int:
        shape = 1, self.in_channels, self.img_size, self.img_size
        with eval_context(self):
            net = self.forward(torch.zeros(*shape, device=get_device(self)))[0]
        return net.shape[1]

    @staticmethod
    def _flatten(net: Tensor, determinate: bool) -> Tuple[Tensor, Any]:
        c, h, w = net.shape[1:]
        if determinate:
            c, h, w = map(int, [c, h, w])
        net = net.view(-1, c, h * w).transpose(1, 2).contiguous()
        return net, (h, w)


@ImgToPatches.register("vanilla")
class VanillaPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        **conv_kwargs: Any,
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        if img_size % patch_size != 0:
            raise ValueError(
                f"`img_size` ({img_size}) should be "
                f"divisible by `patch_size` ({patch_size})"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.projection = Conv2d(
            in_channels,
            latent_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            **conv_kwargs,
        )

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.projection(net)
        return self._flatten(net, determinate)


@ImgToPatches.register("overlap")
class OverlapPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 768,
        stride: int = 4,
        **conv_kwargs: Any,
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        self.conv = Conv2d(
            in_channels,
            latent_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2),
            **conv_kwargs,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.conv(net)
        net, hw = self._flatten(net, determinate)
        net = self.norm(net)
        return net, hw


@ImgToPatches.register("conv")
class ConvPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_channels: int = 64,
        latent_dim: int = 384,
        padding: Optional[int] = None,
        stride: Optional[int] = None,
        bias: bool = False,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        latent_channels_list = [latent_channels] * (num_layers - 1)
        num_channels_list = [in_channels] + latent_channels_list + [latent_dim]
        if padding is None:
            padding = max(1, patch_size // 2)
        if stride is None:
            stride = max(1, (patch_size // 2) - 1)
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    *get_conv_blocks(
                        num_channels_list[i],
                        num_channels_list[i + 1],
                        patch_size,
                        stride,
                        bias=bias,
                        activation=activation,
                        padding=padding,
                    ),
                    nn.MaxPool2d(3, 2, 1),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.conv(net)
        return self._flatten(net, determinate)


__all__ = [
    "PreNorm",
    "ChannelPadding",
    "ImgToPatches",
    "VanillaPatchEmbed",
]
