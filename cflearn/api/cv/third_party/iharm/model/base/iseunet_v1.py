import torch
import torch.nn as nn

from ..ops import MaskedChannelAttention
from ..modeling.unet import UNetDecoder
from ..modeling.unet import UNetEncoder


class ISEUNetV1(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d,
        batchnorm_from=2,
        attend_from=3,
        image_fusion=False,
        ch=64,
        max_channels=512,
        backbone_from=-1,
        backbone_channels=None,
        backbone_mode="",
    ):
        super(ISEUNetV1, self).__init__()
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
            attention_layer=MaskedChannelAttention,
            attend_from=attend_from,
            image_fusion=image_fusion,
        )

    def forward(self, image, mask, backbone_features=None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        output = self.decoder(intermediates, image, mask)
        return {"images": output}
