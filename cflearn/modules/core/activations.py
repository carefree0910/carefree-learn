import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from functools import partial
from torch.nn import Module

from .hijacks import HijackLinear
from ..common import TModule
from ..common import Lambda
from ..common import PrefixModules


activations = PrefixModules("activation")


def register_activation(name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
    return activations.register(name, **kwargs)


def build_activation(
    name: Optional[str],
    config: Optional[Dict[str, Any]] = None,
) -> Module:
    if name is None:
        return nn.Identity()
    if config is None:
        config = {}
    if name.startswith("leaky_relu"):
        splits = name.split("_")
        if len(splits) == 3:
            config["negative_slope"] = float(splits[-1])
        config.setdefault("inplace", True)
        return nn.LeakyReLU(**config)
    if name.lower() == "relu":
        name = "ReLU"
        config.setdefault("inplace", True)
    if activations.has(name):
        return activations.build(name, config=config)
    nn_activation_base = getattr(nn, name, None)
    if nn_activation_base is not None:
        return nn_activation_base(**config)
    func = getattr(torch, name, getattr(F, name, None))
    if func is None:
        msg = f"neither pytorch nor custom activation implemented activation '{name}'"
        raise NotImplementedError(msg)
    return Lambda(partial(func, **config), name)


@register_activation("glu")
class GLU(Module):
    def __init__(self, *, in_dim: int, bias: bool = True):
        super().__init__()
        self.linear = HijackLinear(in_dim, 2 * in_dim, bias)

    def forward(self, net: Tensor) -> Tensor:
        projection, gate = self.linear(net).chunk(2, dim=1)
        return projection * torch.sigmoid(gate)


@register_activation("mish")
class Mish(Module):
    def forward(self, net: Tensor) -> Tensor:
        return net * (torch.tanh(F.softplus(net)))


@register_activation("atanh")
class ATanh(Module):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.atanh(torch.clamp(net, -1.0 + self.eps, 1.0 - self.eps))


@register_activation("isoftplus")
class InverseSoftplus(Module):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.log(net.clamp_min(self.eps).exp() - 1.0)


@register_activation("sign")
class Sign(Module):
    def __init__(
        self,
        *,
        eps: float = 1.0e-12,
        randomize_at_zero: bool = False,
        differentiable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.differentiable = differentiable
        self.randomize_at_zero = randomize_at_zero

    def forward(self, net: Tensor) -> Tensor:
        if self.randomize_at_zero:
            net = net + (2 * torch.empty_like(net).uniform_() - 1.0) * self.eps
        sign = torch.sign(net)
        if not self.differentiable:
            return sign
        return net + (sign - net).detach()


@register_activation("one_hot")
class OneHot(Module):
    def __init__(self, *, differentiable: bool = True):
        super().__init__()
        self.differentiable = differentiable

    def forward(self, net: Tensor) -> Tensor:
        maxed = torch.max(net, dim=1, keepdim=True)[0]
        one_hot = net * (net == maxed).to(torch.float32)
        if not self.differentiable:
            return one_hot
        return net + (one_hot - net).detach()


@register_activation("sine")
class Sine(Module):
    def __init__(self, *, w: float = 1.0):
        super().__init__()
        self.w = w

    def forward(self, net: Tensor) -> Tensor:
        return torch.sin(self.w * net)


@register_activation("h_swish")
class HSwish(Module):
    def __init__(self, *, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, net: Tensor) -> Tensor:
        return net * (self.relu(net + 3.0) / 6.0)


@register_activation("quick_gelu")
class QuickGELU(Module):
    def forward(self, net: Tensor) -> Tensor:
        return net * torch.sigmoid(1.702 * net)


@register_activation("geglu")
class GEGLU(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = HijackLinear(in_dim, out_dim * 2)

    def forward(self, net: Tensor) -> Tensor:
        net, gate = self.net(net).chunk(2, dim=-1)
        return net * F.gelu(gate)


@register_activation("diff_relu")
class DiffReLU(Module):
    def forward(self, net: Tensor) -> Tensor:
        return net + (torch.relu(net) - net).detach()


__all__ = [
    "register_activation",
    "build_activation",
]
