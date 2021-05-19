import torch

from torch import nn
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from cflearn.types import tensor_dict_type
from torchvision.models import vgg16
from torchvision.models import vgg19
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import resnet152
from torchvision.models import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from .....constants import LATENT_KEY
from .....misc.toolkit import set_requires_grad


class VGG(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name == "vgg16":
            make = vgg16
            slices = [4, 9, 16, 23]
        elif name == "vgg19":
            make = vgg19
            slices = [4, 9, 18, 27]
        else:
            raise NotImplementedError(f"'{name}' is not implemented")
        vgg_layers = make(pretrained=pretrained).features
        start_idx = 0
        sliced_modules = []
        for slice_idx in slices:
            local_module = nn.Sequential()
            for i in range(start_idx, slice_idx):
                local_module.add_module(str(i), vgg_layers[i])
            sliced_modules.append(local_module)
            start_idx = slice_idx
        self.slice1 = sliced_modules[0]
        self.slice2 = sliced_modules[1]
        self.slice3 = sliced_modules[2]
        self.slice4 = sliced_modules[3]

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        net = self.slice1(net)
        net = self.slice2(net)
        net = self.slice3(net)
        net = self.slice4(net)
        return net


def sliced_vgg16(pretrained: bool = True) -> VGG:
    return VGG("vgg16", pretrained=pretrained)


def sliced_vgg19(pretrained: bool = True) -> VGG:
    return VGG("vgg19", pretrained=pretrained)


class MobileNet(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        if name == "mobilenet_v2":
            make = mobilenet_v2
            slices = [2, 4, 7, 14, 18]
        else:
            raise NotImplementedError(f"'{name}' is not implemented")
        mobilenet_layers = make(pretrained=pretrained).features
        start_idx = 0
        sliced_modules = []
        for slice_idx in slices:
            local_module = nn.Sequential()
            for i in range(start_idx, slice_idx):
                local_module.add_module(str(i), mobilenet_layers[i])
            sliced_modules.append(local_module)
            start_idx = slice_idx
        self.slice1 = sliced_modules[0]
        self.slice2 = sliced_modules[1]
        self.slice3 = sliced_modules[2]
        self.slice4 = sliced_modules[3]
        self.slice5 = sliced_modules[4]

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        net = self.slice1(net)
        net = self.slice2(net)
        net = self.slice3(net)
        net = self.slice4(net)
        net = self.slice5(net)
        return net


def sliced_mobilenet_v2(pretrained: bool = True) -> MobileNet:
    return MobileNet("mobilenet_v2", pretrained=pretrained)


backbone_dict = {
    "vgg16": sliced_vgg16,
    "vgg19": sliced_vgg19,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "mobilenet_v2": sliced_mobilenet_v2,
}


class Backbone(nn.Module):
    def __init__(
        self,
        name: str = "resnet101",
        *,
        latent_dim: int,
        pretrained: bool = True,
        requires_gap: bool = False,
        need_normalize: bool = False,
        requires_grad: bool = False,
        remove_layers: Optional[List[str]] = None,
        target_layers: Optional[Dict[str, str]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.need_normalize = need_normalize
        self.remove_layers = remove_layers
        self.increment_config = increment_config
        # backbone
        backbone_base = backbone_dict.get(name)
        if backbone_base is None:
            raise ValueError(f"backbone '{name}' is not recognized")
        backbone = backbone_base(pretrained, **kwargs)
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
        # gap
        self.gap = None
        if requires_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # normalize
        if need_normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

    def normalize(self, net: torch.Tensor) -> torch.Tensor:
        return (net - self.mean) / self.std

    def forward(self, net: torch.Tensor) -> tensor_dict_type:
        if self.need_normalize:
            net = self.normalize(net)
        rs = self.core(net)
        net = rs if self._backbone_return_tensor else list(rs.values())[-1]
        if self.gap is not None:
            net = self.gap(net)
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
        super().load_state_dict(state_dict, **kwargs)


__all__ = [
    "Backbone",
]
