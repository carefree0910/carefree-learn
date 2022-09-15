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
    resample: Any = Image.LANCZOS,
    normalize: bool = True,
) -> ReadImageResponse:
    if to_rgb is None:
        raise ValueError("`carefree-cv` is needed for `DiffusionAPI`")
    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("L") if to_gray else to_rgb(image)
    original_w, original_h = image.size
    w, h = restrict_wh(original_w, original_h, max_wh)
    w, h = map(get_suitable_size, (w, h), (anchor, anchor))
    image = image.resize((w, h), resample=resample)
    image = np.array(image)
    if normalize:
        image = image.astype(np.float32) / 255.0
    if to_gray:
        image = image[None, None]
    else:
        image = image[None].transpose(0, 3, 1, 2)
    return ReadImageResponse(image, (original_w, original_h))


T = TypeVar("T", bound="APIMixin")


class APIMixin:
    m: nn.Module
    use_amp: bool

    def __init__(self, m: nn.Module, *, use_amp: bool = False):
        self.m = m
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
        return cls(m.model.core, use_amp=use_amp)
