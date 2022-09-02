import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Optional
from torch.nn import Module
from cftool.misc import shallow_copy_dict

from .basic import get_conv_blocks
from ..utils import Residual
from ..activations import Swish


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


class ResidualBlockWithTimeEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False,
        time_embedding_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = use_conv_shortcut

        make_norm = lambda nc: nn.GroupNorm(num_groups=32, num_channels=nc, eps=1.0e-6)

        self.swish = Swish()
        self.norm1 = make_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        if time_embedding_channels > 0:
            self.time_embedding = nn.Linear(time_embedding_channels, out_channels)
        self.norm2 = make_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, net: Tensor, time_net: Optional[Tensor] = None) -> Tensor:
        inp = net
        net = self.norm1(net)
        net = self.swish(net)
        net = self.conv1(net)

        if time_net is not None:
            time_net = self.swish(time_net)
            net = net + self.time_embedding(time_net)[:, :, None, None]

        net = self.norm2(net)
        net = self.swish(net)
        net = self.dropout(net)
        net = self.conv2(net)

        if self.in_channels != self.out_channels:
            inp = self.shortcut(inp)

        return inp + net


__all__ = [
    "ResidualBlock",
    "ResidualBlockV2",
    "ResidualBlockWithTimeEmbedding",
]
