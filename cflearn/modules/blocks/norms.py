import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from torch.nn import Module
from cftool.misc import update_dict

from ...misc.toolkit import adain_with_params


class BN(nn.BatchNorm1d):
    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


class LN(nn.LayerNorm):
    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) != 4 or len(self.normalized_shape) != 1:
            return super().forward(net)
        batch_size = net.shape[0]
        if batch_size == 1:
            mean = net.mean().view(1, 1, 1, 1)
            std = net.std().view(1, 1, 1, 1)
        else:
            mean = net.view(batch_size, -1).mean(1).view(batch_size, 1, 1, 1)
            std = net.view(batch_size, -1).std(1).view(batch_size, 1, 1, 1)
        net = (net - mean) / (std + self.eps)
        if self.elementwise_affine:
            w = self.weight.view(-1, 1, 1)
            b = self.bias.view(-1, 1, 1)
            net = net * w + b
        return net


class PixelNorm(Module):
    def forward(self, net: Tensor) -> Tensor:
        return F.normalize(net, dim=1)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5, momentum: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None

    def forward(self, net: Tensor) -> Tensor:
        return adain_with_params(net, self.bias, self.weight)

    def extra_repr(self) -> str:
        return str(self.dim)


class NormFactory:
    def __init__(self, norm_type: Optional[str]):
        self.norm_type = norm_type

    @property
    def use_bias(self) -> bool:
        return self.norm_type is None or not self.norm_type.startswith("batch")

    @property
    def norm_base(self) -> Callable:
        norm_type = self.norm_type
        norm_layer: Union[Type[Module], Any]
        if norm_type == "batch_norm":
            norm_layer = BN
        elif norm_type == "layer_norm":
            norm_layer = LN
        elif norm_type == "adain":
            norm_layer = AdaptiveInstanceNorm2d
        elif norm_type == "layer":
            norm_layer = nn.LayerNorm
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "batch1d":
            norm_layer = nn.BatchNorm1d
        elif norm_type == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm_type == "spectral":
            norm_layer = torch.nn.utils.spectral_norm
        elif norm_type == "pixel":
            norm_layer = PixelNorm
        elif norm_type is None:

            def norm_layer(_: Any, *__: Any, **___: Any) -> nn.Identity:
                return nn.Identity()

        else:
            msg = f"normalization layer '{norm_type}' is not found"
            raise NotImplementedError(msg)
        return norm_layer

    @property
    def default_config(self) -> Dict[str, Any]:
        norm_type = self.norm_type
        config: Dict[str, Any] = {}
        if norm_type == "batch":
            config = {"affine": True, "track_running_stats": True}
        elif norm_type == "instance":
            config = {"affine": False, "track_running_stats": False}
        elif norm_type == "layer" or norm_type == "layer_norm":
            config = {"eps": 1.0e-6}
        return config

    def make(self, *args: Any, **kwargs: Any) -> Module:
        kwargs = update_dict(kwargs, self.default_config)
        return self.norm_base(*args, **kwargs)

    def inject_to(
        self,
        dim: int,
        norm_kwargs: Dict[str, Any],
        current_blocks: List[Module],
        *subsequent_blocks: Module,
    ) -> None:
        if self.norm_type != "spectral":
            new_block = self.make(dim, **norm_kwargs)
            current_blocks.append(new_block)
        else:
            last_block = current_blocks[-1]
            last_block = self.make(last_block, **norm_kwargs)
            current_blocks[-1] = last_block
        current_blocks.extend(subsequent_blocks)


__all__ = [
    "BN",
    "LN",
    "PixelNorm",
    "AdaptiveInstanceNorm2d",
    "NormFactory",
]
