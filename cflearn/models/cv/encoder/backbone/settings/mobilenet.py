from typing import List
from collections import OrderedDict

from ..api import Preset


remove_layers: List[str] = []
target_layers = OrderedDict(
    slice0="stage0",
    slice1="stage1",
    slice2="stage2",
    slice3="stage3",
    slice4="stage4",
)


@Preset.register_settings()
class MobileNetPreset(Preset):
    remove_layers = {
        "mobilenet_v2": remove_layers,
    }
    target_layers = {
        "mobilenet_v2": target_layers,
    }
    increment_configs = {
        "mobilenet_v2": {"out_channels": [16, 24, 32, 96, 320]},
    }


__all__ = ["MobileNetPreset"]
