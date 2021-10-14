import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Callable
from typing import Optional

from .....modules.blocks import Conv2d


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> Conv2d:
    return Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> Conv2d:
    return Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, net: Tensor) -> Tensor:
        inp = net

        net = self.conv1(net)
        net = self.bn1(net)
        net = self.relu(net)

        net = self.conv2(net)
        net = self.bn2(net)

        if self.downsample is not None:
            inp = self.downsample(inp)
        net = net + inp
        net = self.relu(net)

        return net


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2d(
            3,
            self.num_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            return_indices=True,
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            return_indices=True,
        )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            return_indices=True,
        )
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            return_indices=True,
        )
        self.maxpool5 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            return_indices=True,
        )
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(
            128,
            layers[1],
            stride=1,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            256,
            layers[2],
            stride=1,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            512,
            layers[3],
            stride=1,
            dilate=replace_stride_with_dilation[2],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        num_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.num_channels != num_channels:
            downsample = nn.Sequential(
                conv1x1(self.num_channels, num_channels, stride),
                norm_layer(num_channels),
            )

        layers = [
            BasicBlock(
                self.num_channels,
                num_channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.num_channels = num_channels
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    self.num_channels,
                    num_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, net: Tensor) -> Tensor:
        x1 = self.conv1(net)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1, idx1 = self.maxpool1(x1)

        x2, idx2 = self.maxpool2(x1)
        x2 = self.layer1(x2)

        x3, idx3 = self.maxpool3(x2)
        x3 = self.layer2(x3)

        x4, idx4 = self.maxpool4(x3)
        x4 = self.layer3(x4)

        x5, idx5 = self.maxpool5(x4)
        x5 = self.layer4(x5)

        x_cls = self.avgpool(x5)
        x_cls = torch.flatten(x_cls, 1)
        x_cls = self.fc(x_cls)

        return x_cls


def resnet34_mp(**kwargs: Any) -> ResNet:
    return ResNet([3, 4, 6, 3], **kwargs)


__all__ = ["resnet34_mp"]
