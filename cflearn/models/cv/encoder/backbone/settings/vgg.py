from typing import List

from ..api import Preset


remove_layers: List[str] = []
target_layers = {
    "slice1": "stage1",
    "slice2": "stage2",
    "slice3": "stage3",
    "slice4": "stage4",
}


@Preset.register_settings()
class VGGPreset(Preset):
    remove_layers = {
        "vgg16": remove_layers,
        "vgg19": remove_layers,
    }
    target_layers = {
        "vgg16": target_layers,
        "vgg19": target_layers,
    }
    latent_dims = {
        "vgg16": -1,
        "vgg19": -1,
    }


__all__ = ["VGGPreset"]
