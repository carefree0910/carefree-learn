from typing import List
from collections import OrderedDict

from ..api import Preset


remove_layers: List[str] = []
target_layers = OrderedDict(
    slice1="stage1",
    slice2="stage2",
    slice3="stage3",
    slice4="stage4",
)


@Preset.register_settings()
class VGGPreset(Preset):
    remove_layers = {
        "vgg16": remove_layers,
        "vgg19": remove_layers,
        "rep_vgg": [],
    }
    target_layers = {
        "vgg16": target_layers,
        "vgg19": target_layers,
        "rep_vgg": OrderedDict(
            stage0="stage0",
            stage1="stage1",
            stage2="stage2",
            stage3_first="stage3_first",
            stage3_second="stage3_second",
            stage4="stage4",
        ),
    }
    increment_configs = {
        "vgg16": {"out_channels": [64, 128, 256, 512]},
        "vgg19": {"out_channels": [64, 128, 256, 512]},
        "rep_vgg": {"out_channels": [64, 160, 320, 640, 640, 2560]},
    }


__all__ = ["VGGPreset"]
