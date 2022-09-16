import torch

import numpy as np
import torch.nn as nn

from PIL import Image
from typing import Any
from typing import Type
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Optional
from typing import NamedTuple
from torch.cuda.amp.autocast_mode import autocast

from ....pipeline import DLPipeline

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


def restrict_wh(w: int, h: int, max_wh: int) -> Tuple[int, int]:
    max_original_wh = max(w, h)
    if max_original_wh <= max_wh:
        return w, h
    wh_ratio = w / h
    if wh_ratio >= 1:
        return max_wh, round(max_wh / wh_ratio)
    return round(max_wh * wh_ratio), max_wh


def get_suitable_size(n: int, anchor: int) -> int:
    mod = n % anchor
    return n - mod + int(mod > 0.5 * anchor) * anchor


class ReadImageResponse(NamedTuple):
    image: np.ndarray
    original_size: Tuple[int, int]


def read_image(
    image: Union[str, Image.Image],
    max_wh: int,
    *,
    anchor: int,
    to_gray: bool = False,
    to_mask: bool = False,
    resample: Any = Image.LANCZOS,
    normalize: bool = True,
) -> ReadImageResponse:
    if to_rgb is None:
        raise ValueError("`carefree-cv` is needed for `DiffusionAPI`")
    if isinstance(image, str):
        image = Image.open(image)
    if not to_mask and not to_gray:
        image = to_rgb(image)
    else:
        if to_mask and to_gray:
            raise ValueError("`to_mask` & `to_gray` should not be True simultaneously")
        if to_mask and image.mode == "RGBA":
            image = image.split()[3]
        else:
            image = image.convert("L")
    original_w, original_h = image.size
    w, h = restrict_wh(original_w, original_h, max_wh)
    w, h = map(get_suitable_size, (w, h), (anchor, anchor))
    image = image.resize((w, h), resample=resample)
    image = np.array(image)
    if normalize:
        image = image.astype(np.float32) / 255.0
    if to_mask or to_gray:
        image = image[None, None]
    else:
        image = image[None].transpose(0, 3, 1, 2)
    return ReadImageResponse(image, (original_w, original_h))


T = TypeVar("T", bound="APIMixin")


class APIMixin:
    m: nn.Module
    device: torch.device
    use_amp: bool

    def __init__(self, m: nn.Module, device: torch.device, *, use_amp: bool = False):
        self.m = m
        self.device = device
        self.use_amp = use_amp

    @property
    def amp_context(self) -> autocast:
        return autocast(enabled=self.use_amp)

    @classmethod
    def from_pipeline(
        cls: Type[T],
        m: DLPipeline,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
    ) -> T:
        if device is not None:
            m.model.to(device)
        return cls(m.model.core, m.model.device, use_amp=use_amp)
