import torch

from torch import nn
from torch import Tensor

from ..misc.toolkit import download_model
from ..misc.toolkit import set_requires_grad
from ..modules.blocks import HijackConv2d


class ScalingLayer(nn.Module):
    mean: Tensor
    std: Tensor

    def __init__(self) -> None:
        super().__init__()
        mean = torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        std = torch.tensor([0.458, 0.448, 0.450])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, net: Tensor) -> Tensor:
        return (net - self.mean) / self.std


# map input channels to 1
class ChannelMapping(nn.Module):
    def __init__(self, in_channels: int, use_dropout: bool):
        super().__init__()
        blocks = []
        if use_dropout:
            blocks.append(nn.Dropout())
        blocks.append(HijackConv2d(in_channels, 1, 1, 1, 0, bias=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


def normalize(net: Tensor, eps: float = 1.0e-10) -> Tensor:
    norm = torch.sqrt(torch.sum(net**2, dim=1, keepdim=True))
    return net / (norm + eps)


def spatial_average(net: Tensor, keepdim: bool = True) -> Tensor:
    return net.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    def __init__(self, use_dropout: bool = True):
        from ..models.cv.encoder.backbone.core import Backbone

        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.backbone = Backbone("vgg16_full", pretrained=True, requires_grad=False)
        make_mapping = lambda in_nc: ChannelMapping(in_nc, use_dropout)
        self.out_channels = self.backbone.out_channels
        self.mappings = nn.ModuleList(list(map(make_mapping, self.out_channels)))
        self.load_pretrained()
        set_requires_grad(self, False)

    def load_pretrained(self) -> None:
        ckpt = download_model("lpips")
        self.load_state_dict(torch.load(ckpt), strict=False)

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        net0, net1 = map(self.scaling_layer, [predictions, target])
        out0, out1 = map(self.backbone, [net0, net1])
        loss = None
        for i in range(len(self.out_channels)):
            stage = f"stage{i}"
            f0, f1 = out0[stage], out1[stage]
            f0, f1 = map(normalize, [f0, f1])
            diff = (f0 - f1) ** 2
            squeezed = self.mappings[i](diff)
            i_loss = spatial_average(squeezed, keepdim=True)
            if loss is None:
                loss = i_loss
            else:
                loss += i_loss
        return loss


__all__ = [
    "LPIPS",
]
