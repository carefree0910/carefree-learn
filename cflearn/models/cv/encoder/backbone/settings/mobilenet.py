from ..api import Preset


remove_layers = []
target_layers = {
    "slice1": "stage0",
    "slice2": "stage1",
    "slice3": "stage2",
    "slice4": "stage3",
    "slice5": "stage4",
}


@Preset.register_settings()
class MobileNetPreset(Preset):
    remove_layers = {
        "mobilenet_v2": remove_layers,
    }
    target_layers = {
        "mobilenet_v2": target_layers,
    }
    latent_dims = {
        "mobilenet_v2": 320,
    }
    increment_configs = {
        "mobilenet_v2": {"out_channels": [16, 24, 32, 96, 320]},
    }


__all__ = ["MobileNetPreset"]
