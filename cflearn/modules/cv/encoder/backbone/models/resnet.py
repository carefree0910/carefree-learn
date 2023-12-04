from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152

from ..register import register_backbone


register_backbone(
    "resnet18",
    [64, 64, 128, 256, 512],
    {
        "relu": "stage0",
        "layer1": "stage1",
        "layer2": "stage2",
        "layer3": "stage3",
        "layer4": "stage4",
    },
)(resnet18)


register_backbone(
    "resnet50",
    [64, 256, 512, 1024, 2048],
    {
        "relu": "stage0",
        "layer1": "stage1",
        "layer2": "stage2",
        "layer3": "stage3",
        "layer4": "stage4",
    },
)(resnet50)


register_backbone(
    "resnet101",
    [64, 256, 512, 1024, 2048],
    {
        "relu": "stage0",
        "layer1": "stage1",
        "layer2": "stage2",
        "layer3": "stage3",
        "layer4": "stage4",
    },
)(resnet101)


register_backbone(
    "resnet152",
    [64, 256, 512, 1024, 2048],
    {
        "relu": "stage0",
        "layer1": "stage1",
        "layer2": "stage2",
        "layer3": "stage3",
        "layer4": "stage4",
    },
)(resnet152)


__all__ = []  # type: ignore
