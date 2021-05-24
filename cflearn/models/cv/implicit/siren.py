import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable
from typing import Optional

from ....types import tensor_dict_type
from ....constants import LABEL_KEY
from ....modules.blocks import Lambda
from ....modules.blocks import Activations
from ....modules.blocks import ChannelPadding


class Sine(nn.Module):
    def __init__(self, w: float = 1.0):
        super().__init__()
        self.w = w

    def forward(self, net: Tensor) -> Tensor:
        return torch.sin(self.w * net)


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        w_sin: float = 1.0,
        c_init: float = 6.0,
        bias: bool = True,
        is_first: bool = False,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.is_first = is_first
        # siren weight
        weight = torch.zeros(out_dim, in_dim)
        w_std = (1.0 / in_dim) if is_first else (math.sqrt(c_init / in_dim) / w_sin)
        weight.uniform_(-w_std, w_std)
        self.weight = nn.Parameter(weight)
        # siren bias
        self.bias = None
        if bias:
            bias_value = torch.zeros(out_dim).uniform_(-w_std, w_std)
            self.bias = nn.Parameter(bias_value)
        # siren activation
        self.activation = Sine(w_sin) if activation is None else activation

    def forward(self, net: Tensor) -> Tensor:
        net = F.linear(net, self.weight, self.bias)
        net = self.activation(net)
        return net


class Modulator(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            dim = in_dim if i == 0 else (latent_dim + in_dim)
            blocks.append(
                nn.Sequential(
                    nn.Linear(dim, latent_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, z: Tensor) -> Tensor:
        net = z
        nets = []
        for block in self.blocks:
            net = block(net)
            nets.append(net)
            net = torch.cat((net, z), dim=1)
        return tuple(nets)


def _make_grid(size: int, in_dim: int) -> Tensor:
    tensors = [torch.linspace(-1.0, 1.0, steps=size)] * in_dim
    grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return grid.view(1, -1, grid.shape[-1])


class Siren(nn.Module):
    def __init__(
        self,
        size: int,
        in_dim: int,
        out_dim: int,
        latent_dim: int,
        *,
        num_layers: int = 4,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        bias: bool = True,
        final_activation: Optional[str] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        # siren blocks
        blocks = []
        for i in range(num_layers):
            is_first = i == 0
            i_w_sin = w_sin_initial if is_first else w_sin
            i_in_dim = in_dim if is_first else latent_dim
            blocks.append(
                SirenLayer(
                    in_dim=i_in_dim,
                    out_dim=latent_dim,
                    w_sin=i_w_sin,
                    bias=bias,
                    is_first=is_first,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        # siren grid
        grid = _make_grid(size, in_dim)
        self.register_buffer("grid", grid)
        # latent modulator
        self.modulator = Modulator(latent_dim, latent_dim, num_layers)
        # head
        self.head = SirenLayer(
            in_dim=latent_dim,
            out_dim=out_dim,
            w_sin=w_sin,
            bias=bias,
            activation=Activations.make(final_activation),
        )

    def forward(self, latent: Tensor, *, size: Optional[int] = None) -> Tensor:
        mods = self.modulator(latent)
        grid = self.grid if size is None else _make_grid(size, self.in_dim)
        grid = grid.to(latent.device)
        net = grid.clone().detach().requires_grad_()
        net = net.repeat(latent.shape[0], 1, 1)
        for block, mod in zip(self.blocks, mods):
            net = block(net) * mod.unsqueeze(1)
        return self.head(net)


# image usages

def img_siren_head(size: int, out_channels: int) -> Callable[[Tensor], Tensor]:
    def _head(t: Tensor) -> Tensor:
        t = t.view(-1, size, size, out_channels)
        t = t.permute(0, 3, 1, 2)
        return t

    return _head


class ImgSiren(nn.Module):
    def __init__(
        self,
        img_size: int,
        out_channels: int,
        latent_dim: int = 256,
        num_classes: Optional[int] = None,
        conditional_dim: int = 16,
        *,
        num_layers: int = 4,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        bias: bool = True,
        final_activation: Optional[str] = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        # condition
        self.cond_padding = None
        if num_classes is not None:
            self.cond_padding = ChannelPadding(conditional_dim, num_classes=num_classes)
            latent_dim += conditional_dim
        # siren
        self.siren = Siren(
            img_size,
            2,
            out_channels,
            latent_dim,
            num_layers=num_layers,
            w_sin=w_sin,
            w_sin_initial=w_sin_initial,
            bias=bias,
            final_activation=final_activation,
        )
        # head
        self.head = Lambda(img_siren_head(img_size, out_channels), name="head")

    def forward(self, net: Tensor, batch: tensor_dict_type) -> Tensor:
        if self.cond_padding is not None:
            net = self.cond_padding(net, batch[LABEL_KEY].view(-1))
        net = self.siren(net)
        return self.head(net)

    def decode(
        self,
        z: Tensor,
        *,
        labels: Optional[Tensor],
        size: Optional[int] = None,
    ) -> Tensor:
        if self.cond_padding is not None:
            if labels is None:
                msg = "`labels` should be provided in conditional `ImgSiren`"
                raise ValueError(msg)
            z = self.cond_padding(z, labels)
        net = self.siren(z, size=size)
        if size is None:
            return self.head(net)
        return img_siren_head(size, self.out_channels)(net)


__all__ = [
    "Siren",
    "ImgSiren",
    "img_siren_head",
]
