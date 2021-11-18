from torchvision.models import mobilenet_v2

from ..register import register_backbone


register_backbone(
    "mobilenet_v2",
    [16, 24, 32, 96, 320],
    {
        "features.1.conv.2": "stage0",
        "features.3.conv.3": "stage1",
        "features.6.conv.3": "stage2",
        "features.13.conv.3": "stage3",
        "features.17.conv.3": "stage4",
    },
)(mobilenet_v2)


__all__ = []  # type: ignore
