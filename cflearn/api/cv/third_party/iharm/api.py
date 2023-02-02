import torch

import numpy as np

from .inference.utils import load_model
from .inference.predictor import Predictor
from .....misc.toolkit import download_model


class ImageHarmonizationAPI:
    def __init__(self, device: torch.device) -> None:
        m = load_model("hrnet32_idih256", download_model("hrnet"))
        self.device = device
        self.predictor = Predictor(m, device, with_flip=False)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.predictor.net.to(device)

    # return uint8 image array
    def run(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.predictor.predict(image, mask)


__all__ = [
    "ImageHarmonizationAPI",
]
