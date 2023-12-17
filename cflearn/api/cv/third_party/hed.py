import os
import torch

import numpy as np
import torch.nn as nn

from ....toolkit import download_checkpoint
from ....parameters import OPT

try:
    import cv2
except:
    cv2 = None
try:
    from einops import rearrange
except:
    rearrange = None


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for _ in range(1, layer_number):
            self.convs.append(
                nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(
            input_channel=3, output_channel=64, layer_number=2
        )
        self.block2 = DoubleConvBlock(
            input_channel=64, output_channel=128, layer_number=2
        )
        self.block3 = DoubleConvBlock(
            input_channel=128, output_channel=256, layer_number=3
        )
        self.block4 = DoubleConvBlock(
            input_channel=256, output_channel=512, layer_number=3
        )
        self.block5 = DoubleConvBlock(
            input_channel=512, output_channel=512, layer_number=3
        )

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HedAPI:
    def __init__(self, device: torch.device):
        if cv2 is None:
            raise RuntimeError("`cv2` is needed for `HedAPI`")
        if rearrange is None:
            raise RuntimeError("`einops` is needed for `HedAPI`")
        self.model = ControlNetHED_Apache2()
        model_path = download_checkpoint("ControlNetHED")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.device = device

    @property
    def dtype(self) -> torch.dtype:
        return list(self.model.parameters())[0].dtype

    def to(self, device: torch.device, *, use_half: bool) -> None:
        if use_half:
            self.model.half()
        else:
            self.model.float()
        self.device = device
        self.model.to(device)

    def __call__(self, input_image: np.ndarray) -> np.ndarray:
        assert input_image.ndim == 3
        H, W, _ = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image.copy()).to(self.device, self.dtype)
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.model(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge


__all__ = [
    "HedAPI",
]
