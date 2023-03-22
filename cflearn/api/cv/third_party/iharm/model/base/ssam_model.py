import torch
from functools import partial

from torch import nn as nn

from ..ops import ChannelAttention
from ..modeling.unet import UNetDecoder
from ..modeling.unet import UNetEncoder
from ..modeling.basic_blocks import ConvBlock
from ..modeling.basic_blocks import GaussianSmoothing


class SSAMImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d,
        batchnorm_from=2,
        attend_from=3,
        attention_mid_k=2.0,
        image_fusion=False,
        ch=64,
        max_channels=512,
        backbone_from=-1,
        backbone_channels=None,
        backbone_mode="",
    ):
        super(SSAMImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = UNetEncoder(
            depth,
            ch,
            norm_layer,
            batchnorm_from,
            max_channels,
            backbone_from,
            backbone_channels,
            backbone_mode,
        )
        self.decoder = UNetDecoder(
            depth,
            self.encoder.block_channels,
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion,
        )

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {"images": output}


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(
            nn.functional.interpolate(
                mask, size=x.size()[-2:], mode="bilinear", align_corners=True
            )
        )
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
