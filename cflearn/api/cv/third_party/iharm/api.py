import numpy as np

from typing import Union
from PIL.Image import Image
from cftool.cv import read_image

from .inference.utils import load_model
from .inference.predictor import Predictor
from ....common import IAPI
from .....schema import device_type
from .....toolkit import download_checkpoint


class ImageHarmonizationAPI(IAPI):
    def __init__(
        self,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        force_not_lazy: bool = False
    ):
        super().__init__(
            load_model("hrnet32_idih256", download_checkpoint("hrnet")),
            device,
            use_amp=use_amp,
            use_half=use_half,
            force_not_lazy=force_not_lazy,
        )
        self.predictor = Predictor(self.m, with_flip=False)

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
        return self.predictor.predict(image_arr, mask_arr, self.dtype, self.device)


__all__ = [
    "ImageHarmonizationAPI",
]
