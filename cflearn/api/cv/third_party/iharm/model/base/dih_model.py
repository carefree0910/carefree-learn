import torch
import torch.nn as nn

from ..modeling.conv_autoencoder import ConvEncoder
from ..modeling.conv_autoencoder import DeconvDecoder


class DeepImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d,
        batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64,
        max_channels=512,
        backbone_from=-1,
        backbone_channels=None,
        backbone_mode="",
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth,
            ch,
            norm_layer,
            batchnorm_from,
            max_channels,
            backbone_from,
            backbone_channels,
            backbone_mode,
        )
        self.decoder = DeconvDecoder(
            depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion
        )

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {"images": output}
