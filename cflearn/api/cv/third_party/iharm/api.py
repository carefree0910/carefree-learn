import numpy as np

from typing import Union
from PIL.Image import Image
from cftool.cv import read_image

from .inference.utils import load_model
from .inference.predictor import Predictor
from .....toolkit import download_checkpoint


class ImageHarmonizationAPI:
    def __init__(self, device: str = "cpu", *, use_half: bool = False) -> None:
        m = load_model("hrnet32_idih256", download_checkpoint("hrnet"))
        self.device = device
        self.predictor = Predictor(m, device, with_flip=False)

    def to(self, device: str, *, use_half: bool = False) -> None:
        self.device = device
        self.predictor.to(device)

    # return uint8 image array
    def predict(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.predictor.predict(image, mask)

    # return uint8 image array
    def run(self, image: Union[str, Image], mask: Union[str, Image]) -> np.ndarray:
        image_arr = read_image(
            image,
            None,
            anchor=None,
            normalize=False,
            to_torch_fmt=False,
        ).image
        mask_arr = read_image(
            mask,
            None,
            anchor=None,
            to_mask=True,
            to_torch_fmt=False,
        ).image
        return self.predict(image_arr, mask_arr)


__all__ = [
    "ImageHarmonizationAPI",
]
