import torch

from torch import nn
from typing import Any
from typing import Dict
from typing import List
from cflearn.types import tensor_dict_type
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models._utils import IntermediateLayerGetter

from .models import *
from .....constants import LATENT_KEY
from .....misc.toolkit import set_requires_grad


backbone_dict = {
    "vgg16": sliced_vgg16,
    "vgg19": sliced_vgg19,
    "rep_vgg": rep_vgg,
    "rep_vgg_lite": rep_vgg_lite,
    "rep_vgg_large": rep_vgg_large,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "mobilenet_v2": sliced_mobilenet_v2,
    "mix_vit": mix_vit,
    "mix_vit_lite": mix_vit_lite,
    "mix_vit_large": mix_vit_large,
}


class Backbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet101",
        *,
        latent_channels: int,
        remove_layers: List[str],
        target_layers: Dict[str, str],
        increment_config: Dict[str, Any],
        pretrained: bool = True,
        requires_grad: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.remove_layers = remove_layers
        self.target_layers = target_layers
        self.increment_config = increment_config
        # backbone
        backbone_base = backbone_dict.get(name)
        if backbone_base is None:
            raise ValueError(f"backbone '{name}' is not recognized")
        backbone = self.raw_backbone = backbone_base(pretrained, **kwargs)
        self._backbone_return_tensor = True
        # layers
        if remove_layers is not None:
            for layer in remove_layers:
                delattr(backbone, layer)
        if target_layers is not None:
            backbone = IntermediateLayerGetter(backbone, target_layers)
            self._backbone_return_tensor = False
        self.core = backbone
        # requires grad
        set_requires_grad(self.core, requires_grad)

    def forward(self, net: torch.Tensor) -> tensor_dict_type:
        rs = self.core(net)
        net = rs if self._backbone_return_tensor else list(rs.values())[-1]
        if self._backbone_return_tensor:
            return {LATENT_KEY: net}
        rs[LATENT_KEY] = net
        return rs

    def load_state_dict(self, state_dict: tensor_dict_type, **kwargs: Any) -> None:  # type: ignore
        pop_keys = []
        if self.remove_layers is not None:
            for key in state_dict:
                for layer in self.remove_layers:
                    if key.startswith(f"{layer}."):
                        pop_keys.append(key)
            for key in pop_keys:
                state_dict.pop(key)
        super().load_state_dict(state_dict, **kwargs)  # type: ignore


__all__ = [
    "Backbone",
    "backbone_dict",
]
