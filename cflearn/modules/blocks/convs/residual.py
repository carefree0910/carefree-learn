import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Optional
from torch.nn import Module
from cftool.misc import shallow_copy_dict

from .basic import conv_nd
from .basic import get_conv_blocks
from ..utils import avg_pool_nd
from ..utils import zero_module
from ..utils import Residual
from ..hijacks import HijackConv2d
from ..hijacks import HijackLinear
from ....misc.toolkit import safe_clip_
from ....misc.toolkit import gradient_checkpoint


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


class ResDownsample(Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool,
        *,
        signal_dim: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        stride = 2 if signal_dim != 3 else (1, 2, 2)
        if not use_conv:
            if in_channels != out_channels:
                raise ValueError(
                    "`in_channels` should be equal to `out_channels` "
                    "when `use_conv` is set to False"
                )
            self.net = avg_pool_nd(signal_dim, kernel_size=stride, stride=stride)
        else:
            self.net = conv_nd(
                signal_dim,
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=padding,
            )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class ResUpsample(Module):
    def __init__(
        self,
        in_channels: int,
        use_conv: bool,
        *,
        signal_dim: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        if not use_conv:
            self.conv = None
        else:
            self.conv = conv_nd(
                signal_dim,
                in_channels,
                out_channels,
                3,
                padding=padding,
            )

    def forward(self, net: Tensor) -> Tensor:
        if self.signal_dim == 3:
            _, _, c, h, w = net.shape
            net = F.interpolate(net, (c, h * 2, w * 2), mode="nearest")
        else:
            net = F.interpolate(net, scale_factor=2, mode="nearest")
        if self.conv is not None:
            net = self.conv(net)
        return net


class ResidualBlockWithTimeEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        signal_dim: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        use_conv_shortcut: bool = False,
        integrate_upsample: bool = False,
        integrate_downsample: bool = False,
        time_embedding_channels: int = 512,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.resample = integrate_upsample or integrate_downsample
        if not self.resample:
            self.inp_resample = self.net_resample = None
        elif integrate_upsample:
            self.inp_resample = ResUpsample(in_channels, False, signal_dim=signal_dim)
            self.net_resample = ResUpsample(in_channels, False, signal_dim=signal_dim)
        else:
            self.inp_resample = ResDownsample(in_channels, False, signal_dim=signal_dim)
            self.net_resample = ResDownsample(in_channels, False, signal_dim=signal_dim)

        make_norm = lambda c: nn.GroupNorm(num_groups=32, num_channels=c, eps=norm_eps)

        self.activation = nn.SiLU()
        self.norm1 = make_norm(in_channels)
        self.conv1 = conv_nd(signal_dim, in_channels, out_channels, 3, 1, 1)
        if time_embedding_channels > 0:
            if use_scale_shift_norm:
                t_out_channels = 2 * out_channels
            else:
                t_out_channels = out_channels
            self.time_embedding = HijackLinear(time_embedding_channels, t_out_channels)
        self.norm2 = make_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        conv2 = conv_nd(signal_dim, out_channels, out_channels, 3, 1, 1)
        self.conv2 = zero_module(conv2)
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = HijackConv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.shortcut = HijackConv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, net: Tensor, time_net: Optional[Tensor] = None) -> Tensor:
        inputs = (net,) if time_net is None else (net, time_net)
        return gradient_checkpoint(
            self._forward,
            inputs=inputs,
            params=self.parameters(),
            enabled=self.use_checkpoint,
        )

    def _forward(self, net: Tensor, time_net: Optional[Tensor] = None) -> Tensor:
        inp = net
        net = self.norm1(net)
        net = self.activation(net)
        if self.inp_resample is None or self.net_resample is None:
            net = self.conv1(net)
        else:
            inp = self.inp_resample(inp)
            net = self.net_resample(net)
            net = self.conv1(net)
        if self.in_channels != self.out_channels:
            inp = self.shortcut(inp)

        if time_net is not None:
            time_net = self.activation(time_net)
            time_net = self.time_embedding(time_net)
            while len(time_net.shape) < len(net.shape):
                time_net = time_net[..., None]
            if self.use_scale_shift_norm:
                scale, shift = torch.chunk(time_net, 2, dim=1)
                net = self.norm2(net) * (1.0 + scale) + shift
                net = self.activation(net)
                net = self.dropout(net)
                net = self.conv2(net)
                return inp + net
            net = net + time_net

        net = self.norm2(net)
        net = self.activation(net)
        net = self.dropout(net)
        net = self.conv2(net)

        net = inp + net
        safe_clip_(net)

        return net


__all__ = [
    "ResidualBlock",
    "ResidualBlockV2",
    "ResidualBlockWithTimeEmbedding",
]
