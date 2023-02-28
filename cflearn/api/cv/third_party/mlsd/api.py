import os
import torch

import numpy as np

from .utils import pred_lines
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .....misc.toolkit import download_model

try:
    import cv2
except:
    cv2 = None


class MLSDDetector:
    def __init__(self, device: torch.device):
        if cv2 is None:
            raise ValueError("`cv2` is needed for `MLSDdetector`")
        model_path = download_model("mlsd_large_512_fp32")
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.to(device).eval()
        self.device = device
        self.use_half = False

    def to(self, device: torch.device, *, use_half: bool) -> None:
        if use_half:
            self.model.half()
        else:
            self.model.float()
        self.model.to(device)
        self.device = device
        self.use_half = use_half

    def __call__(self, input_image, value_threshold: float, distance_threshold: float):
        def to(tensor):
            tensor = tensor.to(self.device)
            tensor = tensor.half() if self.use_half else tensor.float()
            return tensor

        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(
                    img,
                    self.model,
                    to,
                    [img.shape[0], img.shape[1]],
                    value_threshold,
                    distance_threshold,
                )
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(
                        img_output,
                        (x_start, y_start),
                        (x_end, y_end),
                        [255, 255, 255],
                        1,
                    )
        except Exception as e:
            pass
        return img_output[:, :, 0]


__all__ = ["MLSDDetector"]
