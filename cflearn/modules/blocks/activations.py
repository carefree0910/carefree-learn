import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from functools import partial
from torch.nn import Module
from cftool.misc import WithRegister

from .common import Lambda
from .hijacks import HijackLinear


activations: Dict[str, Type["Activation"]] = {}


class Activation(WithRegister["Activation"], Module, metaclass=ABCMeta):
    d = activations

    def __init__(self, **kwargs: Any):
        super().__init__()

    @classmethod
    def make(
        cls,
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
        base = cls.d.get(name, getattr(nn, name, None))
        if base is not None:
            return base(**config)
        func = getattr(torch, name, getattr(F, name, None))
        if func is None:
            raise NotImplementedError(
                "neither pytorch nor custom Activation "
                f"implemented activation '{name}'"
            )
        return Lambda(partial(func, **config), name)


@Activation.register("glu")
class GLU(Activation):
    def __init__(self, *, in_dim: int, bias: bool = True):
        super().__init__()
        self.linear = HijackLinear(in_dim, 2 * in_dim, bias)

    def forward(self, net: Tensor) -> Tensor:
        projection, gate = self.linear(net).chunk(2, dim=1)
        return projection * torch.sigmoid(gate)


@Activation.register("mish")
class Mish(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net * (torch.tanh(F.softplus(net)))


@Activation.register("atanh")
class ATanh(Activation):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.atanh(torch.clamp(net, -1.0 + self.eps, 1.0 - self.eps))


@Activation.register("isoftplus")
class InverseSoftplus(Activation):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.log(net.clamp_min(self.eps).exp() - 1.0)


@Activation.register("sign")
class Sign(Activation):
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


@Activation.register("one_hot")
class OneHot(Activation):
    def __init__(self, *, differentiable: bool = True):
        super().__init__()
        self.differentiable = differentiable

    def forward(self, net: Tensor) -> Tensor:
        maxed = torch.max(net, dim=1, keepdim=True)[0]
        one_hot = net * (net == maxed).to(torch.float32)
        if not self.differentiable:
            return one_hot
        return net + (one_hot - net).detach()


@Activation.register("sine")
class Sine(Activation):
    def __init__(self, *, w: float = 1.0):
        super().__init__()
        self.w = w

    def forward(self, net: Tensor) -> Tensor:
        return torch.sin(self.w * net)


@Activation.register("h_swish")
class HSwish(Activation):
    def __init__(self, *, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, net: Tensor) -> Tensor:
        return net * (self.relu(net + 3.0) / 6.0)


@Activation.register("quick_gelu")
class QuickGELU(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net * torch.sigmoid(1.702 * net)


@Activation.register("geglu")
class GEGLU(Activation):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = HijackLinear(in_dim, out_dim * 2)

    def forward(self, net: Tensor) -> Tensor:
        net, gate = self.net(net).chunk(2, dim=-1)
        return net * F.gelu(gate)


@Activation.register("diff_relu")
class DiffReLU(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net + (torch.relu(net) - net).detach()


__all__ = [
    "Activation",
]
