import torch

from torch import nn
from torchvision.models import mobilenet_v2

from .register import register_backbone


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
        self.num_slices = len(sliced_modules)
        for i, sliced_m in enumerate(sliced_modules):
            setattr(self, f"slice{i}", sliced_m)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_slices):
            net = getattr(self, f"slice{i}")(net)
        return net


@register_backbone("mobilenet_v2")
def sliced_mobilenet_v2(pretrained: bool = True) -> MobileNet:
    return MobileNet("mobilenet_v2", pretrained=pretrained)


__all__ = [
    "sliced_mobilenet_v2",
]
