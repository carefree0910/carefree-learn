from typing import List
from collections import OrderedDict

from ..api import Preset


remove_layers: List[str] = []
target_layers = OrderedDict(
    slice0="stage0",
    slice1="stage1",
    slice2="stage2",
    slice3="stage3",
)

vgg19_large_target_layers = OrderedDict(
    slice0="stage0",
    slice1="stage1",
    slice2="stage2",
    slice3="stage3_first",
    slice4="stage3_second",
    slice5="stage4",
)

rep_vgg_remove_layers: List[str] = []
rep_vgg_target_layers = OrderedDict(
    stage1="stage1",
    stage2="stage2",
    stage3="stage3",
    stage4_first="stage4_first",
    stage4_second="stage4_second",
    stage5="stage5",
)


@Preset.register_settings()
class VGGPreset(Preset):
    remove_layers = {
        "vgg16": remove_layers,
        "vgg19": remove_layers,
        "vgg19_lite": remove_layers,
        "vgg19_large": remove_layers,
        "vgg_style": remove_layers,
        "rep_vgg": rep_vgg_remove_layers,
        "rep_vgg_lite": rep_vgg_remove_layers,
        "rep_vgg_large": rep_vgg_remove_layers,
    }
    target_layers = {
        "vgg16": target_layers,
        "vgg19": target_layers,
        "vgg19_lite": target_layers,
        "vgg19_large": vgg19_large_target_layers,
        "vgg_style": target_layers,
        "rep_vgg": rep_vgg_target_layers,
        "rep_vgg_lite": rep_vgg_target_layers,
        "rep_vgg_large": rep_vgg_target_layers,
    }
    increment_configs = {
        "vgg16": {"out_channels": [64, 128, 256, 512]},
        "vgg19": {"out_channels": [64, 128, 256, 512]},
        "vgg19_lite": {"out_channels": [64, 128, 256, 512]},
        "vgg19_large": {"out_channels": [64, 128, 256, 512, 512, 512]},
        "vgg_style": {"out_channels": [64, 128, 256, 512]},
        "rep_vgg": {"out_channels": [64, 128, 256, 512, 512, 2048]},
        "rep_vgg_lite": {"out_channels": [48, 48, 96, 192, 192, 1280]},
        "rep_vgg_large": {"out_channels": [64, 160, 320, 640, 640, 2560]},
    }


__all__ = ["VGGPreset"]
