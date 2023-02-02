import torch.nn as nn

from .ih_model import IHModelWithBackbone
from ..modifiers import LRMult
from ..modeling.hrnet_ocr import HighResolutionNet
from ..modeling.basic_blocks import MaxPoolDownSize


class HRNetIHModel(IHModelWithBackbone):
    def __init__(
        self,
        base_config,
        downsize_hrnet_input=False,
        mask_fusion="sum",
        lr_mult=0.1,
        cat_hrnet_outputs=True,
        pyramid_channels=-1,
        ocr=64,
        width=18,
        small=True,
        mode="cat",
        **base_kwargs
    ):
        """
        Creates image harmonization model supported by the features extracted from the pre-trained HRNet backbone.
        HRNet outputs feature maps on 4 different resolutions.

        Parameters
        ----------
        base_config : dict
            Configuration dict for the base model, to which the backbone features are incorporated.
            base_config contains model class and init parameters, examples can be found in iharm.mconfigs.base_models
        downsize_backbone_input : bool
            If the input image should be half-sized for the backbone.
        mask_fusion : str
            How to fuse the binary mask with the backbone input:
            'sum': apply convolution to the mask and sum it with the output of the first convolution in the backbone
            'rgb': concatenate the mask to the input image and translate it back to 3 channels with convolution
            otherwise: do not fuse mask with the backbone input
        lr_mult : float
            Multiply learning rate to lr_mult when updating the weights of the backbone.
        cat_hrnet_outputs : bool
            If 4 HRNet outputs should be resized and concatenated to a single tensor.
        pyramid_channels : int
            When HRNet outputs are concatenated to a single one, it can be consequently downsized
            to produce a feature pyramid.
            The pyramid features are then fused with the encoder outputs in the base model on multiple layers.
            Each pyramid feature map contains equal number of channels equal to pyramid_channels.
            If pyramid_channels <= 0, the feature pyramid is not constructed.
        ocr : int
            When HRNet outputs are concatenated to a single one, the OCR module can be applied
            resulting in feature map with (2 * ocr) channels. If ocr <= 0 the OCR module is not applied.
        width : int
            Width of the HRNet blocks.
        small : bool
            If True, HRNet contains 2 blocks at each stage and 4 otherwise.
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

        backbone = HRNetBB(
            cat_outputs=cat_hrnet_outputs,
            pyramid_channels=pyramid_channels,
            pyramid_depth=min(depth - 2 if not downsize_hrnet_input else depth - 3, 4),
            width=width,
            ocr=ocr,
            small=small,
            lr_mult=lr_mult,
        )

        params.update(
            dict(
                backbone_from=3 if downsize_hrnet_input else 2,
                backbone_channels=backbone.output_channels,
                backbone_mode=mode,
            )
        )
        base_model = base_config["model"](**params)

        super(HRNetIHModel, self).__init__(
            base_model, backbone, downsize_hrnet_input, mask_fusion
        )


class HRNetBB(nn.Module):
    def __init__(
        self,
        cat_outputs=True,
        pyramid_channels=256,
        pyramid_depth=4,
        width=18,
        ocr=64,
        small=True,
        lr_mult=0.1,
    ):
        super(HRNetBB, self).__init__()
        self.cat_outputs = cat_outputs
        self.ocr_on = ocr > 0 and cat_outputs
        self.pyramid_on = pyramid_channels > 0 and cat_outputs

        self.hrnet = HighResolutionNet(width, 2, ocr_width=ocr, small=small)
        self.hrnet.apply(LRMult(lr_mult))
        if self.ocr_on:
            self.hrnet.ocr_distri_head.apply(LRMult(1.0))
            self.hrnet.ocr_gather_head.apply(LRMult(1.0))
            self.hrnet.conv3x3_ocr.apply(LRMult(1.0))

        hrnet_cat_channels = [width * 2**i for i in range(4)]
        if self.pyramid_on:
            self.output_channels = [pyramid_channels] * 4
        elif self.ocr_on:
            self.output_channels = [ocr * 2]
        elif self.cat_outputs:
            self.output_channels = [sum(hrnet_cat_channels)]
        else:
            self.output_channels = hrnet_cat_channels

        if self.pyramid_on:
            downsize_in_channels = ocr * 2 if self.ocr_on else sum(hrnet_cat_channels)
            self.downsize = MaxPoolDownSize(
                downsize_in_channels, pyramid_channels, pyramid_channels, pyramid_depth
            )

    def forward(self, image, mask, mask_features):
        if not self.cat_outputs:
            return self.hrnet.compute_hrnet_feats(
                image, mask_features, return_list=True
            )

        outputs = list(self.hrnet(image, mask, mask_features))
        if self.pyramid_on:
            outputs = self.downsize(outputs[0])
        return outputs

    def load_pretrained_weights(self, pretrained_path):
        self.hrnet.load_pretrained_weights(pretrained_path)
