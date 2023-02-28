import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from .....misc.toolkit import download_model


class OpenposeDetector:
    def __init__(self, device: torch.device):
        body_modelpath = download_model("body_pose_model")
        hand_modelpath = download_model("hand_pose_model")

        self.device = device
        self.use_half = False
        self.body_estimation = Body(body_modelpath, device)
        self.hand_estimation = Hand(hand_modelpath, device)

    def to(self, device: torch.device, *, use_half: bool) -> None:
        if use_half:
            self.body_estimation.model.half()
            self.hand_estimation.model.half()
        else:
            self.body_estimation.model.float()
            self.hand_estimation.model.float()
        self.body_estimation.model.to(device)
        self.hand_estimation.model.to(device)
        self.device = device
        self.use_half = use_half

    def __call__(self, input_image, hand=False):
        def to(tensor):
            tensor = tensor.to(self.device)
            tensor = tensor.half() if self.use_half else tensor.float()
            return tensor

        input_image = input_image[..., ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(input_image, to)
            canvas = np.zeros_like(input_image)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, input_image)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(
                        input_image[y : y + w, x : x + w, :], to
                    )
                    peaks[:, 0] = np.where(
                        peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x
                    )
                    peaks[:, 1] = np.where(
                        peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y
                    )
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
            return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())


__all__ = ["OpenposeDetector"]
