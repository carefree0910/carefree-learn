import os
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from cftool.cv import to_uint8

from ....parameters import OPT

try:
    import cv2
except:
    cv2 = None

nets = {
    "carv4": {
        "layer0": "cd",
        "layer1": "ad",
        "layer2": "rd",
        "layer3": "cv",
        "layer4": "cd",
        "layer5": "ad",
        "layer6": "rd",
        "layer7": "cv",
        "layer8": "cd",
        "layer9": "ad",
        "layer10": "rd",
        "layer11": "cv",
        "layer12": "cd",
        "layer13": "ad",
        "layer14": "rd",
        "layer15": "cv",
    },
}


def createConvFunc(op_type):
    assert op_type in ["cv", "cd", "ad", "rd"], "unknown op type: %s" % str(op_type)
    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for cd_conv should be in 1 or 2"
            assert (
                weights.size(2) == 3 and weights.size(3) == 3
            ), "kernel size for cd_conv should be 3x3"
            assert padding == dilation, "padding for cd_conv set wrong"

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(
                x,
                weights,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            return y - yc

        return func
    elif op_type == "ad":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert (
                weights.size(2) == 3 and weights.size(3) == 3
            ), "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(
                shape
            )  # clock-wise
            y = F.conv2d(
                x,
                weights_conv,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            return y

        return func
    elif op_type == "rd":

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for rd_conv should be in 1 or 2"
            assert (
                weights.size(2) == 3 and weights.size(3) == 3
            ), "kernel size for rd_conv should be 3x3"
            padding = 2 * dilation

            shape = weights.shape
            buffer = torch.zeros(
                shape[0],
                shape[1],
                5 * 5,
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(
                x,
                buffer,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            return y

        return func
    else:
        print("impossible to be here unless you force that")
        return None


class Conv2d(nn.Module):
    def __init__(
        self,
        pdc,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False
        )
        self.conv2_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False
        )
        self.conv2_3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False
        )
        self.conv2_4 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            dilation=11,
            padding=11,
            bias=False,
        )
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride

        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(
            pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PDCBlock_converted(nn.Module):
    """
    CPDC, APDC can be converted to vanilla 3x3 convolution
    RPDC can be converted to vanilla 5x5 convolution
    """

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == "rd":
            self.conv1 = nn.Conv2d(
                inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False
            )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y


class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), "dil should be an int"
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane
        if convert:
            if pdcs[0] == "rd":
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(
                3,
                self.inplane,
                kernel_size=init_kernel_size,
                padding=init_padding,
                bias=False,
            )
            block_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock

        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.attentions.append(CSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(CSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        # print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if "bn" in pname:
                bn_weights.append(p)
            elif "relu" in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        outputs = [e1, e2, e3, e4]

        output = self.classifier(torch.cat(outputs, dim=1))
        # if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, "unrecognized model, please choose from %s" % str(
        model_options
    )

    # print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = "layer%d" % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs


def pidinet():
    pdcs = config_model("carv4")
    return PiDiNet(60, pdcs, dil=24, sa=True)


class PiDiAPI:
    def __init__(self, device: torch.device):
        if cv2 is None:
            raise RuntimeError("`cv2` is needed for `PiDiAPI`")
        model_path = os.path.join(OPT.external_dir, "annotators", "table5_pidinet.pth")
        if not os.path.isfile(model_path):
            raise ValueError(f"cannot find `table5_pidinet.pth` at {model_path}")
        self.model = pidinet()
        d = torch.load(model_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in d.items()})
        self.device = device

    @property
    def dtype(self) -> torch.dtype:
        return list(self.model.parameters())[0].dtype

    def to(self, device: torch.device, *, use_half: bool) -> None:
        if use_half:
            self.model.half()
        else:
            self.model.float()
        self.device = device
        self.model.to(device)

    def __call__(self, uint8_rgb: np.ndarray, threshold: Optional[float]) -> np.ndarray:
        net = torch.from_numpy(uint8_rgb[None]).to(self.device, dtype=self.dtype)
        net = net.permute(0, 3, 1, 2).contiguous()
        net /= 255.0
        res = self.model(net)[-1]
        if threshold is not None:
            res = res > threshold
        res = res[0, 0].float().cpu().data.numpy()
        return to_uint8(res)


__all__ = [
    "PiDiAPI",
]
