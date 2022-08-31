import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Optional
from torch.nn import Module
from cftool.misc import shallow_copy_dict

from .basic import get_conv_blocks
from ..utils import Residual


class ResidualBlock(Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        ca_reduction: Optional[int] = None,
        eca_kernel_size: Optional[int] = None,
        norm_type: Optional[str] = "batch",
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["norm_type"] = norm_type
        k1 = shallow_copy_dict(kwargs)
        k1["ca_reduction"] = ca_reduction
        k1.setdefault("activation", nn.LeakyReLU(0.2, inplace=True))
        blocks = get_conv_blocks(dim, dim, kernel_size, stride, **k1)
        if 0.0 < dropout < 1.0:
            blocks.append(nn.Dropout(dropout))
        k2 = shallow_copy_dict(kwargs)
        k2["activation"] = None
        k2["eca_kernel_size"] = eca_kernel_size
        blocks.extend(get_conv_blocks(dim, dim, kernel_size, stride, **k2))
        self.net = Residual(nn.Sequential(*blocks))

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class ResidualBlockV2(Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        ca_reduction: Optional[int] = None,
        eca_kernel_size: Optional[int] = None,
        norm_type: Optional[str] = "batch",
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["norm_type"] = norm_type
        k1 = shallow_copy_dict(kwargs)
        k1["pre_activate"] = True
        k1["ca_reduction"] = ca_reduction
        k1.setdefault("activation", nn.LeakyReLU(0.2, inplace=True))
        blocks = get_conv_blocks(dim, dim, kernel_size, stride, **k1)
        if 0.0 < dropout < 1.0:
            blocks.append(nn.Dropout(dropout))
        k2 = shallow_copy_dict(kwargs)
        k2["pre_activate"] = True
        k2["eca_kernel_size"] = eca_kernel_size
        blocks.extend(get_conv_blocks(dim, dim, kernel_size, stride, **k2))
        self.net = Residual(nn.Sequential(*blocks))

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "ResidualBlock",
    "ResidualBlockV2",
]
