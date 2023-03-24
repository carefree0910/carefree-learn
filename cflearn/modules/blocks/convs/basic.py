import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from torch.nn import Module
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn.modules.pooling import _MaxUnpoolNd

from ..norms import NormFactory
from ..hijacks import HijackConv1d
from ..hijacks import HijackConv2d
from ..hijacks import HijackConv3d
from ..activations import Activation
from ....misc.toolkit import interpolate


class GaussianBlur3(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        base = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = base[:, None] * base[None, :] / 16.0
        kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.kernel: Tensor
        self.register_buffer("kernel", kernel)
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, groups=self.in_channels, padding=1)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Any = "same",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        weight_scale: Optional[float] = None,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        self.kernel_size = kernel_size
        self.reflection_pad = None
        if padding == "same":
            padding = kernel_size // 2
        elif isinstance(padding, str) and padding.startswith("reflection"):
            reflection_padding: Any
            if padding == "reflection":
                reflection_padding = kernel_size // 2
            else:
                reflection_padding = int(padding[len("reflection") :])
            padding = 0
            if transform_kernel:
                reflection_padding = [reflection_padding] * 4
                reflection_padding[0] += 1
                reflection_padding[2] += 1
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.groups, self.stride = groups, stride
        self.dilation, self.padding = dilation, padding
        self.transform_kernel = transform_kernel
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        if not bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.demodulate = demodulate
        self.weight_scale = weight_scale
        # initialize
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight.data, gain / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.zero_()

    def _same_padding(self, size: int) -> int:
        stride = self.stride
        dilation = self.dilation
        return ((size - 1) * (stride - 1) + dilation * (self.kernel_size - 1)) // 2

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
    ) -> Tensor:
        b, c, *hw = net.shape
        # padding
        padding = self.padding
        if self.padding == "same":
            padding = tuple(map(self._same_padding, hw))
        if self.reflection_pad is not None:
            net = self.reflection_pad(net)
        # transform kernel
        w: Union[nn.Parameter, Tensor] = self.weight
        if self.transform_kernel:
            w = F.pad(w, [1, 1, 1, 1], mode="constant")
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            ) * 0.25
        # ordinary convolution
        if style is None:
            bias = self.bias
            groups = self.groups
        # 'stylized' convolution, used in StyleGAN
        else:
            suffix = "when `style` is provided"
            if self.bias is not None:
                raise ValueError(f"`bias` should not be used {suffix}")
            if self.groups != 1:
                raise ValueError(f"`groups` should be 1 {suffix}")
            if self.reflection_pad is not None:
                raise ValueError(
                    f"`reflection_pad` should not be used {suffix}, "
                    "maybe you want to use `same` padding?"
                )
            w = w[None, ...] * style[..., None, :, None, None]
            # prepare for group convolution
            bias = None
            groups = b
            net = net.view(1, -1, *hw)  # 1, b*in, h, w
            w = w.view(b * self.out_c, *w.shape[2:])  # b*out, in, wh, ww
        if self.demodulate:
            w = w * torch.rsqrt(w.pow(2).sum([-3, -2, -1], keepdim=True) + 1e-8)
        if self.weight_scale is not None:
            w = w * self.weight_scale
        # conv core
        if not transpose:
            fn = F.conv2d
        else:
            fn = F.conv_transpose2d
            if groups == 1:
                w = w.transpose(0, 1)
            else:
                oc, ic, kh, kw = w.shape
                w = w.reshape(groups, oc // groups, ic, kh, kw)
                w = w.transpose(1, 2)
                w = w.reshape(groups * ic, oc // groups, kh, kw)
        net = fn(
            net,
            w,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
        )
        if style is None:
            return net
        return net.view(b, -1, *net.shape[2:])

    def extra_repr(self) -> str:
        return (
            f"{self.in_c}, {self.out_c}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}, demodulate={self.demodulate}"
        )


class DepthWiseConv2d(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(output_size, int):
            output_size = output_size, output_size
        self.h, self.w = output_size

    def forward(self, net: Tensor) -> Tensor:
        h, w = map(int, net.shape[2:])
        sh, sw = map(math.floor, [h / self.h, w / self.w])  # type: ignore
        kh = h - (self.h - 1) * sh  # type: ignore
        kw = w - (self.w - 1) * sw  # type: ignore
        return F.avg_pool2d(net, kernel_size=(kh, kw), stride=(sh, sw))


class MaxUnpool2d_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return F.max_unpool2d(*args)

    @staticmethod
    def symbolic(
        g: Any,
        net: Any,
        indices: Any,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        output_size: Any,
    ) -> Any:
        # get shape
        input_shape = g.op("Shape", net)
        const_0 = g.op("Constant", value_t=torch.tensor(0))
        const_1 = g.op("Constant", value_t=torch.tensor(1))
        batch_size = g.op("Gather", input_shape, const_0, axis_i=0)
        channel = g.op("Gather", input_shape, const_1, axis_i=0)

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            "Gather",
            input_shape,
            g.op("Constant", value_t=torch.tensor(2)),
            axis_i=0,
        )
        height = g.op("Sub", height, const_1)
        height = g.op("Mul", height, g.op("Constant", value_t=torch.tensor(stride[1])))
        height = g.op(
            "Add",
            height,
            g.op("Constant", value_t=torch.tensor(kernel_size[1])),
        )

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            "Gather",
            input_shape,
            g.op("Constant", value_t=torch.tensor(3)),
            axis_i=0,
        )
        width = g.op("Sub", width, const_1)
        width = g.op("Mul", width, g.op("Constant", value_t=torch.tensor(stride[0])))
        width = g.op(
            "Add",
            width,
            g.op("Constant", value_t=torch.tensor(kernel_size[0])),
        )

        # step of channel
        channel_step = g.op("Mul", height, width)
        # step of batch
        batch_step = g.op("Mul", channel_step, channel)

        # channel offset
        range_channel = g.op("Range", const_0, channel, const_1)
        range_channel = g.op(
            "Reshape",
            range_channel,
            g.op("Constant", value_t=torch.tensor([1, -1, 1, 1])),
        )
        range_channel = g.op("Mul", range_channel, channel_step)
        range_channel = g.op("Cast", range_channel, to_i=7)  # 7 is int64

        # batch offset
        range_batch = g.op("Range", const_0, batch_size, const_1)
        range_batch = g.op(
            "Reshape",
            range_batch,
            g.op("Constant", value_t=torch.tensor([-1, 1, 1, 1])),
        )
        range_batch = g.op("Mul", range_batch, batch_step)
        range_batch = g.op("Cast", range_batch, to_i=7)  # 7 is int64

        # update indices
        indices = g.op("Add", indices, range_channel)
        indices = g.op("Add", indices, range_batch)

        return g.op(
            "MaxUnpool",
            net,
            indices,
            kernel_shape_i=kernel_size,
            strides_i=stride,
        )


class MaxUnpool2d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Any, stride: Any = None, padding: Any = 0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, net: Tensor, indices: Tensor, output_size: Any = None) -> Tensor:
        return MaxUnpool2d_op.apply(
            net,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size,
        )


class Interpolate(Module):
    def __init__(self, factor: Optional[float] = None, mode: str = "nearest"):
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.kw = dict(mode=mode, factor=factor)

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tensor:
        if self.factor is not None:
            net = interpolate(net, determinate=determinate, **self.kw)  # type: ignore
        return net

    def extra_repr(self) -> str:
        return f"{self.factor}, {self.mode}"


class UpsampleConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "nearest",
        padding: Optional[Union[int, str]] = None,
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        factor: Optional[float] = None,
    ):
        if mode == "transpose":
            if factor == 1.0:
                mode = "nearest"
                factor = None
            elif factor is not None:
                stride = int(round(factor))
                if padding is None:
                    padding = 0
        if padding is None:
            padding = "same"
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
            transform_kernel=transform_kernel,
            bias=bias,
            demodulate=demodulate,
        )
        if mode == "transpose":
            self.upsample = None
        else:
            self.upsample = Interpolate(factor, mode)

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
        determinate: bool = False,
    ) -> Tensor:
        if self.upsample is None:
            transpose = True
        else:
            net = self.upsample(net, determinate=determinate)
            if transpose:
                raise ValueError("should not use transpose when `upsample` is used")
        return super().forward(net, style, transpose=transpose)


class CABlock(Module):
    """Coordinate Attention"""

    def __init__(self, num_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        latent_channels = max(8, num_channels // reduction)
        self.conv_blocks = nn.Sequential(
            *get_conv_blocks(
                num_channels,
                latent_channels,
                kernel_size=1,
                stride=1,
                norm_type="batch",
                activation=Activation.make("h_swish"),
                padding=0,
            )
        )

        conv2d = lambda: Conv2d(
            latent_channels,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_h = conv2d()
        self.conv_w = conv2d()

    def forward(self, net: Tensor) -> Tensor:
        original = net

        n, c, h, w = net.shape
        net_h = self.pool_h(net)
        net_w = self.pool_w(net).transpose(2, 3)

        net = torch.cat([net_h, net_w], dim=2)
        net = self.conv_blocks(net)

        net_h, net_w = torch.split(net, [h, w], dim=2)
        net_w = net_w.transpose(2, 3)

        net_h = self.conv_h(net_h).sigmoid()
        net_w = self.conv_w(net_w).sigmoid()

        return original * net_w * net_h


class ECABlock(Module):
    """Efficient Channel Attention"""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, net: Tensor) -> Tensor:
        w = self.avg_pool(net).squeeze(-1).transpose(-1, -2)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        return w * net


class SEBlock(Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        impl: str = "conv",
        block_impl: str = "cflearn",
    ):
        super().__init__()
        self.in_channels = in_channels
        if block_impl == "cflearn":
            conv_base = Conv2d
            self.avg_pool = AdaptiveAvgPool2d(1)
        elif block_impl == "torch":
            conv_base = nn.Conv2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"unrecognized `block_impl` ({block_impl}) found")
        self.fc = self.up = self.down = None
        if impl == "conv":
            self.down = conv_base(
                in_channels,
                latent_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            )
            self.up = conv_base(
                latent_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        elif impl == "fc":
            self.fc = nn.Sequential(
                nn.Linear(in_channels, latent_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(latent_channels, in_channels, bias=False),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"implementation '{impl}' is not recognized")

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.avg_pool(net)
        if self.fc is not None:
            net = self.fc(net.view(-1, self.in_channels))
        elif self.up is not None and self.down is not None:
            net = self.down(net)
            net = F.relu(net)
            net = self.up(net)
            net = torch.sigmoid(net)
        net = net.view(-1, self.in_channels, 1, 1)
        return inp * net


def conv_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return HijackConv1d(*args, **kwargs)
    elif n == 2:
        return HijackConv2d(*args, **kwargs)
    elif n == 3:
        return HijackConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


def get_conv_blocks(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    *,
    bias: bool = True,
    demodulate: bool = False,
    norm_type: Optional[str] = None,
    norm_kwargs: Optional[Dict[str, Any]] = None,
    ca_reduction: Optional[int] = None,
    eca_kernel_size: Optional[int] = None,
    activation: Optional[Union[str, Module]] = None,
    conv_base: Type["Conv2d"] = Conv2d,
    pre_activate: bool = False,
    **conv2d_kwargs: Any,
) -> List[Module]:
    conv = conv_base(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        demodulate=demodulate,
        **conv2d_kwargs,
    )
    blocks: List[Module] = []
    if not pre_activate:
        blocks.append(conv)
    if not demodulate:
        factory = NormFactory(norm_type)
        factory.inject_to(out_channels, norm_kwargs or {}, blocks)
    if eca_kernel_size is not None:
        blocks.append(ECABlock(kernel_size))
    if activation is not None:
        if isinstance(activation, str):
            activation = Activation.make(activation)
        blocks.append(activation)
    if ca_reduction is not None:
        blocks.append(CABlock(out_channels, ca_reduction))
    if pre_activate:
        blocks.append(conv)
    return blocks


__all__ = [
    "GaussianBlur3",
    "Conv2d",
    "DepthWiseConv2d",
    "AdaptiveAvgPool2d",
    "MaxUnpool2d",
    "Interpolate",
    "UpsampleConv2d",
    "CABlock",
    "ECABlock",
    "SEBlock",
    "conv_nd",
    "get_conv_blocks",
]
