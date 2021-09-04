from torch import nn
from typing import Any
from typing import Dict
from typing import Callable
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152


backbone_fn_type = Callable[[Any], nn.Module]
backbone_dict: Dict[str, backbone_fn_type] = {}


def register_backbone(name: str) -> Callable[[backbone_fn_type], backbone_fn_type]:
    def _register(f: backbone_fn_type) -> backbone_fn_type:
        backbone_dict[name] = f
        return f

    return _register


register_backbone("resnet18")(resnet18)
register_backbone("resnet50")(resnet50)
register_backbone("resnet101")(resnet101)
register_backbone("resnet152")(resnet152)


__all__ = ["register_backbone"]
