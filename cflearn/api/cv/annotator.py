import torch

import numpy as np

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar
from typing import Optional
from cftool.misc import WithRegister

from .third_party import HedAPI
from .third_party import PiDiAPI
from .third_party import MiDaSAPI
from .third_party import MLSDDetector
from .third_party import OpenposeDetector

try:
    import cv2
except:
    cv2 = None


annotators: Dict[str, Type["Annotator"]] = {}
TAnnotator = TypeVar("TAnnotator", bound="Annotator")


class Annotator(WithRegister):
    d = annotators

    @abstractmethod
    def __init__(self, device: torch.device) -> None:
        pass

    @abstractmethod
    def to(self: TAnnotator, device: torch.device, *, use_half: bool) -> TAnnotator:
        pass

    @abstractmethod
    def annotate(self, uint8_rgb: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass


@Annotator.register("depth")
class DepthAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        self.m = MiDaSAPI(device)

    def to(self, device: torch.device, *, use_half: bool) -> "DepthAnnotator":
        self.m.to(device, use_half=use_half)
        return self

    def annotate(self, uint8_rgb: np.ndarray) -> np.ndarray:  # type: ignore
        return self.m.detect_depth(uint8_rgb)


@Annotator.register("canny")
class CannyAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        if cv2 is None:
            raise RuntimeError("`cv2` is needed for `CannyAnnotator`")

    def to(self, device: torch.device, *, use_half: bool) -> "CannyAnnotator":
        return self

    def annotate(  # type: ignore
        self,
        uint8_rgb: np.ndarray,
        *,
        low_threshold: int,
        high_threshold: int,
    ) -> np.ndarray:
        return cv2.Canny(uint8_rgb, low_threshold, high_threshold)


@Annotator.register("pose")
class PoseAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        self.m = OpenposeDetector(device)

    def to(self, device: torch.device, *, use_half: bool) -> "PoseAnnotator":
        self.m.to(device, use_half=use_half)
        return self

    def annotate(self, uint8_rgb: np.ndarray, hand: bool = False) -> np.ndarray:  # type: ignore
        return self.m(uint8_rgb, hand)[0]


@Annotator.register("mlsd")
class MLSDAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        self.m = MLSDDetector(device)

    def to(self, device: torch.device, *, use_half: bool) -> "MLSDAnnotator":
        self.m.to(device, use_half=use_half)
        return self

    def annotate(  # type: ignore
        self,
        uint8_rgb: np.ndarray,
        *,
        value_threshold: float,
        distance_threshold: float,
    ) -> np.ndarray:
        return self.m(uint8_rgb, value_threshold, distance_threshold)


@Annotator.register("softedge")
class SoftEdgeAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        self.m = HedAPI(device)

    def to(self, device: torch.device, *, use_half: bool) -> "SoftEdgeAnnotator":
        self.m.to(device, use_half=use_half)
        return self

    def annotate(self, uint8_rgb: np.ndarray) -> np.ndarray:  # type: ignore
        return self.m(uint8_rgb)


@Annotator.register("pidi")
class PiDiAnnotator(Annotator):
    def __init__(self, device: torch.device) -> None:
        self.m = PiDiAPI(device)

    def to(self, device: torch.device, *, use_half: bool) -> "PiDiAnnotator":
        self.m.to(device, use_half=use_half)
        return self

    def annotate(  # type: ignore
        self,
        uint8_rgb: np.ndarray,
        *,
        threshold: Optional[float],
    ) -> np.ndarray:  # type: ignore
        return self.m(uint8_rgb, threshold)


__all__ = [
    "Annotator",
    "DepthAnnotator",
    "CannyAnnotator",
    "PoseAnnotator",
    "MLSDAnnotator",
    "SoftEdgeAnnotator",
    "PiDiAnnotator",
]
