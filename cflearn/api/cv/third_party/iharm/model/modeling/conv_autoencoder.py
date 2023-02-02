import torch

import torch.nn as nn

from .basic_blocks import ConvBlock
from ..ops import FeaturesConnector
from ..ops import MaskedChannelAttention


class ConvEncoder(nn.Module):
    def __init__(
        self,
        depth,
        ch,
        norm_layer,
        batchnorm_from,
        max_channels,
        backbone_from,
        backbone_channels=None,
        backbone_mode="",
    ):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        backbone_channels = [] if backbone_channels is None else backbone_channels[::-1]

        in_channels = 4
        out_channels = ch

        self.block0 = ConvBlock(
            in_channels,
            out_channels,
            norm_layer=norm_layer if batchnorm_from == 0 else None,
        )
        self.block1 = ConvBlock(
            out_channels,
            out_channels,
            norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None,
        )
        self.blocks_channels = [out_channels, out_channels]

        self.blocks_connected = nn.ModuleDict()
        self.connectors = nn.ModuleDict()
        for block_i in range(2, depth):
            if block_i % 2:
                in_channels = out_channels
            else:
                in_channels, out_channels = out_channels, min(
                    2 * out_channels, max_channels
                )

            if 0 <= backbone_from <= block_i and len(backbone_channels):
                stage_channels = backbone_channels.pop()
                connector = FeaturesConnector(
                    backbone_mode, in_channels, stage_channels, in_channels
                )
                self.connectors[f"connector{block_i}"] = connector
                in_channels = connector.output_channels

            self.blocks_connected[f"block{block_i}"] = ConvBlock(
                in_channels,
                out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                padding=int(block_i < depth - 1),
            )
            self.blocks_channels += [out_channels]

    def forward(self, x, backbone_features):
        backbone_features = [] if backbone_features is None else backbone_features[::-1]

        outputs = [self.block0(x)]
        outputs += [self.block1(outputs[-1])]

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f"block{block_i}"]
            output = outputs[-1]
            connector_name = f"connector{block_i}"
            if connector_name in self.connectors:
                stage_features = backbone_features.pop()
                connector = self.connectors[connector_name]
                output = connector(output, stage_features)
            outputs += [block(output)]

        return outputs[::-1]


class DeconvDecoder(nn.Module):
    def __init__(
        self,
        depth,
        encoder_blocks_channels,
        norm_layer,
        attend_from=-1,
        image_fusion=False,
    ):
        super(DeconvDecoder, self).__init__()
        self.image_fusion = image_fusion
        self.deconv_blocks = nn.ModuleList()

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth):
            out_channels = (
                encoder_blocks_channels.pop()
                if len(encoder_blocks_channels)
                else in_channels // 2
            )
            self.deconv_blocks.append(
                SEDeconvBlock(
                    in_channels,
                    out_channels,
                    norm_layer=norm_layer,
                    padding=0 if d == 0 else 1,
                    with_se=0 <= attend_from <= d,
                )
            )
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, image, mask=None):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.deconv_blocks[:-1], encoder_outputs[1:]):
            output = block(output, mask)
            output = output + skip_output
        output = self.deconv_blocks[-1](output, mask)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output


class SEDeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ELU,
        with_se=False,
    ):
        super(SEDeconvBlock, self).__init__()
        self.with_se = with_se
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.with_se:
            self.se = MaskedChannelAttention(out_channels)

    def forward(self, x, mask=None):
        out = self.block(x)
        if self.with_se:
            out = self.se(out, mask)
        return out
