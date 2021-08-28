import torch

from torch import nn
from torchvision.models import mobilenet_v2


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


__all__ = [
    "sliced_mobilenet_v2",
]
