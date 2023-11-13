import torch.nn as nn

from .ih_model import IHModelWithBackbone
from ..modifiers import LRMult
from ..modeling.deeplab_v3 import DeepLabV3Plus
from ..modeling.basic_blocks import MaxPoolDownSize


class DeepLabIHModel(IHModelWithBackbone):
    def __init__(
        self,
        base_config,
        mask_fusion="sum",
        deeplab_backbone="resnet34",
        lr_mult=0.1,
        pyramid_channels=-1,
        deeplab_ch=256,
        mode="cat",
        **base_kwargs
    ):
        """
        Creates image harmonization model supported by the features extracted from the pre-trained DeepLab backbone.

        Parameters
        ----------
        base_config : dict
            Configuration dict for the base model, to which the backbone features are incorporated.
            base_config contains model class and init parameters, examples can be found in iharm.mconfigs.base_models
        mask_fusion : str
            How to fuse the binary mask with the backbone input:
            'sum': apply convolution to the mask and sum it with the output of the first convolution in the backbone
            'rgb': concatenate the mask to the input image and translate it back to 3 channels with convolution
            otherwise: do not fuse mask with the backbone input
        deeplab_backbone : str
            ResNet backbone name.
        lr_mult : float
            Multiply learning rate to lr_mult when updating the weights of the backbone.
        pyramid_channels : int
            The DeepLab output can be consequently downsized to produce a feature pyramid.
            The pyramid features are then fused with the encoder outputs in the base model on multiple layers.
            Each pyramid feature map contains equal number of channels equal to pyramid_channels.
            If pyramid_channels <= 0, the feature pyramid is not constructed.
        deeplab_ch : int
            Number of channels for output DeepLab layer and some in the middle.
        mode : str
            How to fuse the backbone features with the encoder outputs in the base model:
            'sum': apply convolution to the backbone feature map obtaining number of channels
             same as in the encoder output and sum them
            'cat': concatenate the backbone feature map with the encoder output
            'catc': concatenate the backbone feature map with the encoder output and apply convolution obtaining
            number of channels same as in the encoder output
            otherwise: the backbone features are not incorporated into the base model
        base_kwargs : dict
            any kwargs associated with the base model
        """
        params = base_config["params"]
        params.update(base_kwargs)
        depth = params["depth"]

        backbone = DeepLabBB(pyramid_channels, deeplab_ch, deeplab_backbone, lr_mult)

        downsize_input = depth > 7
        params.update(
            dict(
                backbone_from=3 if downsize_input else 2,
                backbone_channels=backbone.output_channels,
                backbone_mode=mode,
            )
        )
        base_model = base_config["model"](**params)

        super(DeepLabIHModel, self).__init__(
            base_model, backbone, downsize_input, mask_fusion
        )


class DeepLabBB(nn.Module):
    def __init__(
        self,
        pyramid_channels=256,
        deeplab_ch=256,
        backbone="resnet34",
        backbone_lr_mult=0.1,
    ):
        super(DeepLabBB, self).__init__()
        self.pyramid_on = pyramid_channels > 0
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        else:
            self.output_channels = [deeplab_ch]

        self.deeplab = DeepLabV3Plus(
            backbone=backbone,
            ch=deeplab_ch,
            project_dropout=0.2,
            norm_layer=nn.BatchNorm2d,
            backbone_norm_layer=nn.BatchNorm2d,
        )
        self.deeplab.backbone.apply(LRMult(backbone_lr_mult))

        if self.pyramid_on:
            self.downsize = MaxPoolDownSize(
                deeplab_ch, pyramid_channels, pyramid_channels, 4
            )

    def forward(self, image, mask, mask_features):
        outputs = list(self.deeplab(image, mask_features))
        if self.pyramid_on:
            outputs = self.downsize(outputs[0])
        return outputs

    def load_pretrained_weights(self):
        self.deeplab.load_pretrained_weights()
