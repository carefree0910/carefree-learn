from ..api import Preset


remove_layers = ["fc", "avgpool"]
target_layers = {
    "relu": "stage0",
    "layer1": "stage1",
    "layer2": "stage2",
    "layer3": "stage3",
    "layer4": "stage4",
}


@Preset.register_settings()
class ResNetPreset(Preset):
    remove_layers = {
        "resnet18": remove_layers,
        "resnet50": remove_layers,
        "resnet101": remove_layers,
    }
    target_layers = {
        "resnet18": target_layers,
        "resnet50": target_layers,
        "resnet101": target_layers,
    }
    latent_dims = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
    }
    num_downsample = {
        "resnet18": 5,
        "resnet50": 5,
        "resnet101": 5,
    }
    increment_configs = {
        "resnet18": {"out_channels": [64, 64, 128, 256, 512]},
        "resnet50": {"out_channels": [64, 256, 512, 1024, 2048]},
        "resnet101": {"out_channels": [64, 256, 512, 1024, 2048]},
    }


__all__ = ["ResNetPreset"]
