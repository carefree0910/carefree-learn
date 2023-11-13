import torch

import torch.nn as nn

from ..ops import ScaleLayer
from ..ops import SimpleInputFusion


class IHModelWithBackbone(nn.Module):
    def __init__(
        self,
        model,
        backbone,
        downsize_backbone_input=False,
        mask_fusion="sum",
        backbone_conv1_channels=64,
    ):
        """
        Creates image harmonization model supported by the features extracted from the pre-trained backbone.

        Parameters
        ----------
        model : nn.Module
            Image harmonization model takes image and mask as an input and handles features from the backbone network.
        backbone : nn.Module
            Backbone model accepts RGB image and returns a list of features.
        downsize_backbone_input : bool
            If the input image should be half-sized for the backbone.
        mask_fusion : str
            How to fuse the binary mask with the backbone input:
            'sum': apply convolution to the mask and sum it with the output of the first convolution in the backbone
            'rgb': concatenate the mask to the input image and translate it back to 3 channels with convolution
            otherwise: do not fuse mask with the backbone input
        backbone_conv1_channels : int
            If mask_fusion is 'sum', define the number of channels for the convolution applied to the mask.
        """
        super(IHModelWithBackbone, self).__init__()
        self.downsize_backbone_input = downsize_backbone_input
        self.mask_fusion = mask_fusion

        self.backbone = backbone
        self.model = model

        if mask_fusion == "rgb":
            self.fusion = SimpleInputFusion()
        elif mask_fusion == "sum":
            self.mask_conv = nn.Sequential(
                nn.Conv2d(
                    1,
                    backbone_conv1_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                ScaleLayer(init_value=0.1, lr_mult=1),
            )

    def forward(self, image, mask):
        """
        Forward the backbone model and then the base model, supported by the backbone feature maps.
        Return model predictions.

        Parameters
        ----------
        image : torch.Tensor
            Input RGB image.
        mask : torch.Tensor
            Binary mask of the foreground region.

        Returns
        -------
        torch.Tensor
            Harmonized RGB image.
        """
        backbone_image = image
        backbone_mask = torch.cat((mask, 1.0 - mask), dim=1)
        if self.downsize_backbone_input:
            backbone_image = nn.functional.interpolate(
                backbone_image, scale_factor=0.5, mode="bilinear", align_corners=True
            )
            backbone_mask = nn.functional.interpolate(
                backbone_mask,
                backbone_image.size()[2:],
                mode="bilinear",
                align_corners=True,
            )
        backbone_image = (
            self.fusion(backbone_image, backbone_mask[:, :1])
            if self.mask_fusion == "rgb"
            else backbone_image
        )
        backbone_mask_features = (
            self.mask_conv(backbone_mask[:, :1]) if self.mask_fusion == "sum" else None
        )
        backbone_features = self.backbone(
            backbone_image, backbone_mask, backbone_mask_features
        )

        output = self.model(image, mask, backbone_features)
        return output
