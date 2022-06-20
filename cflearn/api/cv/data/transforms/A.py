from typing import Any
from typing import Tuple
from typing import Union
from typing import Optional

from .general import ToRGB
from .general import ToGray
from .general import BatchWrapper
from .....data import Transforms
from .....constants import INPUT_KEY
from .....constants import LABEL_KEY

try:
    import cv2
except:
    cv2 = None
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except:
    A = ToTensorV2 = None


class ATransforms(Transforms):
    input_alias = "image"

    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__()
        self.label_alias = label_alias
        if A is None:
            msg = f"`albumentations` is needed for `{self.__class__.__name__}`"
            raise ValueError(msg)

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


AToRGB = lambda: BatchWrapper(ToRGB(), ATransforms.input_alias)
AToGray = lambda: BatchWrapper(ToGray(), ATransforms.input_alias)


@Transforms.register("a_resize")
class AResize(ATransforms):
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


@Transforms.register("a_random_crop")
class ARandomCrop(ATransforms):
    def __init__(self, size: Union[int, tuple], *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        if isinstance(size, int):
            size = size, size
        self.fn = A.RandomCrop(*size)


@Transforms.register("a_shift_scale_rotate")
class AShiftScaleRotate(ATransforms):
    def __init__(
        self,
        p: float = 0.5,
        border_mode: Optional[int] = None,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        if border_mode is None:
            if cv2 is None:
                raise ValueError("`cv2` is needed for `AShiftScaleRotate`")
            border_mode = cv2.BORDER_REFLECT_101
        self.fn = A.ShiftScaleRotate(border_mode=border_mode, p=p)


@Transforms.register("a_hflip")
class AHFlip(ATransforms):
    def __init__(self, p: float = 0.5, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = A.HorizontalFlip(p=p)


@Transforms.register("a_vflip")
class AVFlip(ATransforms):
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


@Transforms.register("a_rgb_shift")
class ARGBShift(ATransforms):
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


@Transforms.register("a_solarize")
class ASolarize(ATransforms):
    def __init__(
        self,
        threshold: float = 0.5,
        p: float = 0.5,
        *,
        label_alias: Optional[str] = None,
    ):
        super().__init__(label_alias=label_alias)
        self.fn = A.Solarize(threshold, p=p)


@Transforms.register("a_gaussian_blur")
class AGaussianBlur(ATransforms):
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


@Transforms.register("a_hue_saturation")
class AHueSaturationValue(ATransforms):
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


@Transforms.register("a_brightness_contrast")
class ARandomBrightnessContrast(ATransforms):
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
    "AToRGB",
    "AToGray",
    "AResize",
    "ARandomCrop",
    "AShiftScaleRotate",
    "AHFlip",
    "AVFlip",
    "ANormalize",
    "ARGBShift",
    "ASolarize",
    "AGaussianBlur",
    "AHueSaturationValue",
    "ARandomBrightnessContrast",
    "AToTensor",
    "ATransforms",
]
