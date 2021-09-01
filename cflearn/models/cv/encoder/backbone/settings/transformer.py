from typing import List
from collections import OrderedDict

from ..api import Preset


remove_layers: List[str] = []
target_layers = OrderedDict(
    stage1="stage1",
    stage2="stage2",
    stage3="stage3",
    stage4="stage4",
)


@Preset.register_settings()
class MixViTPreset(Preset):
    remove_layers = {
        "mix_vit": remove_layers,
        "mix_vit_lite": remove_layers,
        "mix_vit_large": remove_layers,
    }
    target_layers = {
        "mix_vit": target_layers,
        "mix_vit_lite": target_layers,
        "mix_vit_large": target_layers,
    }
    increment_configs = {
        "mix_vit": {"out_channels": [64, 128, 320, 512]},
        "mix_vit_lite": {"out_channels": [32, 64, 160, 256]},
        "mix_vit_large": {"out_channels": [64, 128, 320, 512]},
    }


__all__ = ["MixViTPreset"]
