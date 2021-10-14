import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple

from .backbone import resnet34_mp
from ...protocol import ImageTranslatorMixin
from .....misc.toolkit import interpolate
from .....misc.internal_ import register_module
from .....modules.blocks import get_conv_blocks
from .....modules.blocks import Conv2d
from .....modules.blocks import SEBlock
from .....modules.blocks import Interpolate
from .....modules.blocks import MaxUnpool2d
from .....modules.blocks import AdaptiveAvgPool2d


def psp_stage(num_channels: int, size: int) -> nn.Sequential:
    prior = AdaptiveAvgPool2d(output_size=size)
    conv = Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
    return nn.Sequential(prior, conv)


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        stages = [psp_stage(in_channels, size) for size in sizes]
        self.stages = nn.ModuleList(stages)
        self.bottleneck = Conv2d(
            in_channels * (len(sizes) + 1),
            out_channels,
            kernel_size=1,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, net: Tensor, determinate: bool) -> Tensor:
        kw = dict(anchor=net, mode="bilinear", determinate=determinate)
        priors = [interpolate(stage(net), **kw) for stage in self.stages]
        bottle = self.bottleneck(torch.cat(priors + [net], 1))
        return self.relu(bottle)


def make_se(in_channels: int, reduction: int = 16) -> SEBlock:
    return SEBlock(in_channels, in_channels // reduction, impl="fc")


def fuse(global_logits: Tensor, local_logits: Tensor) -> Tensor:
    values, index = torch.max(global_logits, 1)
    index = index[:, None, :, :].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask == 2] = 1
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask == 2] = 0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask == 1] = 0
    fg_mask[fg_mask == 2] = 1
    fusion_sigmoid = torch.sigmoid(local_logits) * trimap_mask + fg_mask
    return fusion_sigmoid


class ConvUpPSP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                Interpolate(factor, "bilinear"),
            ]
        )

    def forward(self, net: Tensor, determinate: bool) -> Tensor:
        for block in self.blocks[:-1]:
            net = block(net)
        return self.blocks[-1](net, determinate=determinate)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        *,
        reduce_first: bool = True,
        reduce_last: bool = True,
        upsample: bool = True,
        padding: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            if i == 0 and reduce_first:
                out_channels = in_channels // 2
            elif i == num_layers - 1 and reduce_last:
                out_channels = in_channels // 2
            else:
                out_channels = in_channels
            blocks.extend(
                get_conv_blocks(
                    in_channels,
                    out_channels,
                    3,
                    stride=1,
                    norm_type="batch",
                    activation="relu",
                    padding=padding,
                    **kwargs,
                )
            )
            in_channels = out_channels
        if upsample:
            blocks.append(Interpolate(2, "bilinear"))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, net: Tensor, determinate: bool) -> Tensor:
        for block in self.blocks:
            kw = {}
            if isinstance(block, Interpolate):
                kw["determinate"] = determinate
            net = block(net, **kw)
        return net


@register_module("aim", pre_bases=[ImageTranslatorMixin])
class AIMNet(nn.Module):
    def __init__(self):  # type: ignore
        super().__init__()
        # encoder
        resnet = resnet34_mp()
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.mp0 = resnet.maxpool1
        self.encoder1 = resnet.layer1
        self.mp1 = resnet.maxpool2
        self.encoder2 = resnet.layer2
        self.mp2 = resnet.maxpool3
        self.encoder3 = resnet.layer3
        self.mp3 = resnet.maxpool4
        self.encoder4 = resnet.layer4
        self.mp4 = resnet.maxpool5
        # decoder
        # global
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp4 = ConvUpPSP(512, 256, 2)
        self.psp3 = ConvUpPSP(512, 128, 4)
        self.psp2 = ConvUpPSP(512, 64, 8)
        self.psp1 = ConvUpPSP(512, 64, 16)
        # stage 4d
        self.decoder4_g = DecoderStage(1024, 3)
        self.decoder4_g_se = make_se(256)
        # stage 3d
        self.decoder3_g = DecoderStage(512, 3)
        self.decoder3_g_se = make_se(128)
        # stage 2d
        self.decoder2_g = DecoderStage(256, 3)
        self.decoder2_g_se = make_se(64)
        # stage 1d
        self.decoder1_g = DecoderStage(128, 3, reduce_last=False)
        self.decoder1_g_se = make_se(64)
        # stage 0d
        self.decoder0_g = DecoderStage(128, 2, reduce_last=False)
        self.decoder0_g_spatial = Conv2d(2, 1, kernel_size=7, padding=3)
        self.decoder0_g_se = make_se(64)
        self.decoder_final_g = Conv2d(64, 3, kernel_size=3, padding=1)
        # local
        self.bridge_block = DecoderStage(
            512,
            3,
            reduce_first=False,
            reduce_last=False,
            upsample=False,
            padding=2,
            dilation=2,
        )
        self.max_unpool = MaxUnpool2d(2, 2)
        # stage 4l
        self.decoder4_l = DecoderStage(1024, 3, upsample=False)
        # stage 3l
        self.decoder3_l = DecoderStage(512, 3, upsample=False)
        # stage 2l
        self.decoder2_l = DecoderStage(256, 3, upsample=False)
        # stage 1l
        self.decoder1_l = DecoderStage(128, 3, reduce_last=False, upsample=False)
        # stage 0l
        self.decoder0_l = DecoderStage(128, 2, reduce_last=False, upsample=False)
        self.decoder_final_l = Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, net: Tensor, *, determinate: bool = False) -> List[Tensor]:
        # encoder
        e0 = self.encoder0(net)
        # e0: N, 64, H, W
        e0p, id0 = self.mp0(e0)
        e1p, id1 = self.mp1(e0p)
        # e1p: N, 64, H/4, W/4
        e1 = self.encoder1(e1p)
        # e1: N, 64, H/2, W/2
        e2p, id2 = self.mp2(e1)
        # e2p: N, 64, H/8, W/8
        e2 = self.encoder2(e2p)
        # e2: N, 128, H/4, W/4
        e3p, id3 = self.mp3(e2)
        # e3p: N, 128, H/16, W/16
        e3 = self.encoder3(e3p)
        # e3: N, 256, H/8, W/8
        e4p, id4 = self.mp4(e3)
        # e4p: N, 256, H/32, W/32
        e4 = self.encoder4(e4p)
        # e4p: N, 512, H/16, W/16

        # decoder
        psp = self.psp_module(e4, determinate)
        # psp: N, 512, H/32, W/32
        d4_g = torch.cat((psp, e4), 1)
        d4_g = self.decoder4_g(d4_g, determinate)
        # d4_g: N, 256, H/16, W/16
        d4_g = self.decoder4_g_se(d4_g)
        d3_g = torch.cat((self.psp4(psp, determinate), d4_g), 1)
        d3_g = self.decoder3_g(d3_g, determinate)
        # d3_g: N, 128, H/8, W/8
        d3_g = self.decoder3_g_se(d3_g)
        d2_g = torch.cat((self.psp3(psp, determinate), d3_g), 1)
        d2_g = self.decoder2_g(d2_g, determinate)
        # d2_g: N, 64, H/4, W/4
        d2_g = self.decoder2_g_se(d2_g)
        d1_g = torch.cat((self.psp2(psp, determinate), d2_g), 1)
        d1_g = self.decoder1_g(d1_g, determinate)
        # d1_g: N, 64, H/2, W/2
        d1_g = self.decoder1_g_se(d1_g)
        d0_g = torch.cat((self.psp1(psp, determinate), d1_g), 1)
        d0_g = self.decoder0_g(d0_g, determinate)
        # d0_g: N, 64, H, W

        d0_g_avg = torch.mean(d0_g, dim=1, keepdim=True)
        d0_g_max, _ = torch.max(d0_g, dim=1, keepdim=True)
        d0_g_cat = torch.cat([d0_g_avg, d0_g_max], dim=1)
        d0_g_spatial = self.decoder0_g_spatial(d0_g_cat)
        d0_g_spatial_sigmoid = torch.sigmoid(d0_g_spatial)

        d0_g = self.decoder0_g_se(d0_g)
        global_logits = self.decoder_final_g(d0_g)

        bb = self.bridge_block(e4, determinate)
        # bb: N, 512, H/32, W/32
        d4_l = self.decoder4_l(torch.cat((bb, e4), 1), determinate)
        # d4_l: N, 256, H/32, W/32
        d3_l = self.max_unpool(d4_l, id4)
        # d3_l: N, 256, H/16, W/16
        d3_l = self.decoder3_l(torch.cat((d3_l, e3), 1), determinate)
        # d3_l: N, 128, H/16, W/16
        d2_l = self.max_unpool(d3_l, id3)
        # d2_l: N, 128, H/8, W/8
        d2_l = self.decoder2_l(torch.cat((d2_l, e2), 1), determinate)
        # d2_l: N, 64, H/8, W/8
        d1_l = self.max_unpool(d2_l, id2)
        # d1_l: N, 64, H/4, W/4
        d1_l = self.decoder1_l(torch.cat((d1_l, e1), 1), determinate)
        # d1_l: N, 64, H/4, W/4
        d0_l = self.max_unpool(d1_l, id1)
        # d0_l: N, 64, H/2, W/2
        d0_l = self.max_unpool(d0_l, id0)
        # d0_l: N, 64, H, W
        d0_l = self.decoder0_l(torch.cat((d0_l, e0), 1), determinate)
        # d0_l: N, 64, H, W
        d0_l = d0_l + d0_l * d0_g_spatial_sigmoid
        local_logits = self.decoder_final_l(d0_l)

        fusion_sigmoid = fuse(global_logits, local_logits)
        return [fusion_sigmoid, global_logits, local_logits]


__all__ = ["AIMNet"]
