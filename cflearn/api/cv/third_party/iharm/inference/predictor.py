import torch

import numpy as np

from torch import nn
from typing import Tuple

from .transforms import ToTensor
from .transforms import PadToDivisor
from .transforms import NormalizeTensor
from .transforms import AddFlippedTensor
from ..model.backboned import HRNetIHModel


class Predictor:
    def __init__(
        self,
        net: nn.Module,
        device: torch.device,
        *,
        with_flip: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()

        if hasattr(net, "depth"):
            size_divisor = 2 ** (net.depth + 1)
        elif isinstance(net, HRNetIHModel):
            size_divisor = 128
        else:
            size_divisor = 1

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        self.transforms = [
            PadToDivisor(divisor=size_divisor, border_mode=0),
            ToTensor("cpu"),
            NormalizeTensor(mean, std, "cpu"),
        ]
        if with_flip:
            self.transforms.append(AddFlippedTensor())

    def to(self, device: torch.device) -> None:
        self.device = device
        self.net.to(device)

    # return uint8 image array
    @torch.no_grad()
    def predict(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image, mask = transform.transform(image, mask)
        image, mask = image.to(self.device), mask.to(self.device)
        predicted_image = self.net(image, mask)["images"].cpu()
        for transform in reversed(self.transforms):
            predicted_image = transform.inv_transform(predicted_image)
        predicted_image = torch.clamp(predicted_image, 0, 255).to(torch.uint8)
        return predicted_image.cpu().numpy()
