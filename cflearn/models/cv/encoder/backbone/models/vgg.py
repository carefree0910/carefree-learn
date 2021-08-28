import torch

from torch import nn
from torchvision.models import vgg16
from torchvision.models import vgg19


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


__all__ = [
    "sliced_vgg16",
    "sliced_vgg19",
]
