import cv2

import numpy as np
import albumentations as A

from typing import Any
from typing import Tuple
from typing import Union
from typing import Optional
from albumentations.pytorch import ToTensorV2

from .protocol import Transforms
from .....types import np_dict_type
from .....constants import INPUT_KEY
from .....constants import LABEL_KEY


class ATransforms(Transforms):
    input_alias = "image"

    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__()
        self.label_alias = label_alias

    def __call__(self, inp: Any, **kwargs: Any) -> Any:  # type: ignore
        if not self.need_batch_process:
            kwargs[self.input_alias] = inp
            return self.fn(**kwargs)[self.input_alias]
        inp_keys_mapping = {
            self.input_alias
            if k == INPUT_KEY
            else self.label_alias
            if k == LABEL_KEY
            else k: k
            for k in inp
        }
        inp = {k: inp[v] for k, v in inp_keys_mapping.items()}
        return {inp_keys_mapping[k]: v for k, v in self.fn(**inp).items()}

    @property
    def need_batch_process(self) -> bool:
        return self.label_alias is not None

    @property
    def need_numpy(self) -> bool:
        return True


@Transforms.register("to_array")
class ToArray(ATransforms):
    @staticmethod
    def fn(**inp: Any) -> np_dict_type:
        return {k: np.array(v) for k, v in inp.items()}


@Transforms.register("to_rgb")
class ToRGB(ATransforms):
    def _to_rgb(self, k: str, v: np.ndarray) -> np.ndarray:
        if k != self.input_alias:
            return v
        if len(v.shape) == 2:
            v = v[..., None]
        if v.shape[2] == 3:
            return v
        if v.shape[2] == 4:
            return v[..., :3] * v[..., 3:]
        if v.shape[2] == 1:
            return v.repeat(3, axis=2)
        raise ValueError(f"invalid shape ({v.shape}) occurred with '{k}'")

    def fn(self, **inp: np.ndarray) -> np_dict_type:
        return {k: self._to_rgb(k, v) for k, v in inp.items()}


@Transforms.register("resize")
class Resize(ATransforms):
    def __init__(
        self,
        size: Union[int, tuple] = 224,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        if isinstance(size, int):
            size = size, size
        self.fn = A.Resize(*size)


@Transforms.register("random_crop")
class RandomCrop(ATransforms):
    def __init__(self, size: Union[int, tuple], *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        if isinstance(size, int):
            size = size, size
        self.fn = A.RandomCrop(*size)


@Transforms.register("shift_scale_rotate")
class ShiftScaleRotate(ATransforms):
    def __init__(
        self,
        p: float = 0.5,
        border_mode: int = cv2.BORDER_REFLECT_101,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.ShiftScaleRotate(border_mode=border_mode, p=p)


@Transforms.register("hflip")
class HFlip(ATransforms):
    def __init__(self, p: float = 0.5, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = A.HorizontalFlip(p=p)


@Transforms.register("vflip")
class VFlip(ATransforms):
    def __init__(self, p: float = 0.5, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = A.VerticalFlip(p=p)


@Transforms.register("a_normalize")
class ANormalize(ATransforms):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 1.0,
        p: float = 1.0,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.Normalize(mean, std, max_pixel_value, p=p)


@Transforms.register("rgb_shift")
class RGBShift(ATransforms):
    def __init__(
        self,
        r_shift_limit: float = 0.08,
        g_shift_limit: float = 0.08,
        b_shift_limit: float = 0.08,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.RGBShift(r_shift_limit, g_shift_limit, b_shift_limit, p=p)


@Transforms.register("solarize")
class Solarize(ATransforms):
    def __init__(
        self,
        threshold: float = 0.5,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.Solarize(threshold, p=p)


@Transforms.register("gaussian_blur")
class GaussianBlur(ATransforms):
    def __init__(
        self,
        blur_limit: Tuple[int, int] = (3, 7),
        sigma_limit: int = 0,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.GaussianBlur(blur_limit, sigma_limit, p=p)


@Transforms.register("hue_saturation")
class HueSaturationValue(ATransforms):
    def __init__(
        self,
        hue_shift_limit: float = 0.08,
        sat_shift_limit: float = 0.12,
        val_shift_limit: float = 0.08,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.HueSaturationValue(
            hue_shift_limit,
            sat_shift_limit,
            val_shift_limit,
            p,
        )


@Transforms.register("brightness_contrast")
class RandomBrightnessContrast(ATransforms):
    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        brightness_by_max: bool = True,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.RandomBrightnessContrast(
            brightness_limit,
            contrast_limit,
            brightness_by_max,
            p,
        )


@Transforms.register("a_to_tensor")
class AToTensor(ATransforms):
    def __init__(
        self,
        transpose_mask: bool = True,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = ToTensorV2(transpose_mask)


__all__ = [
    "ToArray",
    "ToRGB",
    "Resize",
    "RandomCrop",
    "ShiftScaleRotate",
    "HFlip",
    "VFlip",
    "ANormalize",
    "RGBShift",
    "Solarize",
    "GaussianBlur",
    "HueSaturationValue",
    "RandomBrightnessContrast",
    "AToTensor",
]
