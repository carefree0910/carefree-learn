import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch import Tensor
from cftool.cv import to_rgb
from torchvision.transforms.functional import normalize

from ....misc.toolkit import download_model


class REBNCONV(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dilation_rate: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.conv_s1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=dilation_rate,
            dilation=dilation_rate,
            stride=stride,
        )
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, net: Tensor) -> Tensor:
        net = self.relu_s1(self.bn_s1(self.conv_s1(net)))
        return net


def _upsample_like(src: Tensor, tgt: Tensor) -> Tensor:
    net = F.interpolate(src, size=tgt.shape[2:], mode="bilinear")
    return net


class RSU7(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        self.in_ch = in_channels
        self.mid_ch = mid_channels
        self.out_ch = out_channels

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation_rate=1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)

        self.rebnconv7 = REBNCONV(mid_channels, mid_channels, dilation_rate=2)

        self.rebnconv6d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv5d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv4d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv3d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv2d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv1d = REBNCONV(mid_channels * 2, out_channels, dilation_rate=1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation_rate=1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)

        self.rebnconv6 = REBNCONV(mid_channels, mid_channels, dilation_rate=2)

        self.rebnconv5d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv4d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv3d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv2d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv1d = REBNCONV(mid_channels * 2, out_channels, dilation_rate=1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation_rate=1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)

        self.rebnconv5 = REBNCONV(mid_channels, mid_channels, dilation_rate=2)

        self.rebnconv4d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv3d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv2d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv1d = REBNCONV(mid_channels * 2, out_channels, dilation_rate=1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation_rate=1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation_rate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation_rate=1)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation_rate=2)

        self.rebnconv3d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv2d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=1)
        self.rebnconv1d = REBNCONV(mid_channels * 2, out_channels, dilation_rate=1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 12,
        out_channels: int = 3,
    ):
        super().__init__()

        self.rebnconvin = REBNCONV(in_channels, out_channels, dilation_rate=1)

        self.rebnconv1 = REBNCONV(out_channels, mid_channels, dilation_rate=1)
        self.rebnconv2 = REBNCONV(mid_channels, mid_channels, dilation_rate=2)
        self.rebnconv3 = REBNCONV(mid_channels, mid_channels, dilation_rate=4)

        self.rebnconv4 = REBNCONV(mid_channels, mid_channels, dilation_rate=8)

        self.rebnconv3d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=4)
        self.rebnconv2d = REBNCONV(mid_channels * 2, mid_channels, dilation_rate=2)
        self.rebnconv1d = REBNCONV(mid_channels * 2, out_channels, dilation_rate=1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class ISNetDIS(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self, net: Tensor) -> Tensor:
        hx = net

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, net)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, net)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, net)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, net)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, net)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, net)

        # d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return [
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        ], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]


class ISNetAPI:
    def __init__(self, device: torch.device) -> None:
        self.model = ISNetDIS()
        self.device = device
        model_path = download_model("isnet")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)
        self.model.eval()

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def segment(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(to_rgb(image))
        shape = rgb.shape[:2]
        net = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        net = net.permute(2, 0, 1)[None]
        net = F.interpolate(net, (1024, 1024), mode="bilinear")
        net = normalize(net / 255.0, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        net = self.model(net)
        net = F.interpolate(net[0][0], shape, mode="bilinear")
        net_min, net_max = net.min(), net.max()
        net = (net - net_min) / (net_max - net_min)
        return net.cpu().numpy()[0][0]


__all__ = [
    "ISNetAPI",
]
