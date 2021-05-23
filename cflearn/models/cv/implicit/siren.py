import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

from ....modules.blocks import Activations


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


__all__ = ["Siren"]
