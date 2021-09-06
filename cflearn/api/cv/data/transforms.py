import cv2
import random

import numpy as np
import albumentations as A

from abc import abstractmethod
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
from albumentations.pytorch import ToTensorV2

from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....misc.toolkit import WithRegister


cf_transforms: Dict[str, Type["Transforms"]] = {}


class Transforms(WithRegister):
    d: Dict[str, Type["Transforms"]] = cf_transforms

    fn: Any

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __call__(self, inp: Any, *args: Any, **kwargs: Any) -> Any:
        if self.need_batch_process and not isinstance(inp, dict):
            raise ValueError(f"`inp` should be a batch for {self.__class__.__name__}")
        return self.fn(inp, *args, **kwargs)

    @property
    @abstractmethod
    def need_batch_process(self) -> bool:
        pass

    @classmethod
    def convert(
        cls,
        transform: Optional[Union[str, List[str], "Transforms", Callable]],
        transform_config: Optional[Dict[str, Any]] = None,
    ) -> Optional["Transforms"]:
        if transform is None:
            return None
        if isinstance(transform, Transforms):
            return transform
        if callable(transform):
            need_batch = (transform_config or {}).get("need_batch_process", False)
            return Function(transform, need_batch)
        if transform_config is None:
            transform_config = {}
        if isinstance(transform, str):
            return cls.make(transform, transform_config)
        transform_list = [cls.make(t, transform_config.get(t, {})) for t in transform]
        return Compose(transform_list)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "Transforms":
        split = name.split("_")
        if len(split) >= 3 and split[-2] == "with":
            name = "_".join(split[:-2])
            config.setdefault("label_alias", split[-1])
        return super().make(name, config)


class Function(Transforms):
    def __init__(self, fn: Callable, need_batch_process: bool = False):
        super().__init__()
        self.fn = fn
        self._need_batch_process = need_batch_process

    @property
    def need_batch_process(self) -> bool:
        return self._need_batch_process


@Transforms.register("compose")
class Compose(Transforms):
    def __init__(self, transform_list: List[Transforms]):
        super().__init__()
        if len(set(t.need_batch_process for t in transform_list)) > 1:
            raise ValueError(
                "all transforms should have identical "
                "`need_batch_process` property in `Compose`"
            )
        self.fn = transforms.Compose(transform_list)
        self.transform_list = transform_list

    @property
    def need_batch_process(self) -> bool:
        return self.transform_list[0].need_batch_process


@Transforms.register("to_gray")
class ToGray(Transforms):
    fn = transforms.Grayscale()

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("to_tensor")
class ToTensor(Transforms):
    fn = transforms.ToTensor()

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("random_resized_crop")
class RandomResizedCrop(Transforms):
    def __init__(self, *, size: int = 224):
        super().__init__()
        self.fn = transforms.RandomResizedCrop(size)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("-1~1")
class N1To1(Transforms):
    fn = transforms.Lambda(lambda t: t * 2.0 - 1.0)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("inverse_0~1")
class Inverse0To1(Transforms):
    fn = transforms.Lambda(lambda t: 1.0 - t)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("inverse_-1~1")
class InverseN1To1(Transforms):
    fn = transforms.Lambda(lambda t: -t)

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("for_generation")
class ForGeneration(Compose):
    def __init__(self):  # type: ignore
        super().__init__([ToTensor(), N1To1()])


@Transforms.register("for_imagenet")
class ForImagenet(Compose):
    def __init__(self):  # type: ignore
        super().__init__([ToArray(), Resize(224), Normalize(), ToTensor()])


@Transforms.register("ssl")
class SSLTransform(Transforms):
    class Augmentation:
        class GaussianBlur:
            def __init__(
                self,
                p: float = 0.5,
                radius_min: float = 0.1,
                radius_max: float = 2.0,
            ):
                self.prob = p
                self.radius_min = radius_min
                self.radius_max = radius_max

            def __call__(self, img: Image) -> Image:
                if random.random() > self.prob:
                    return img
                r = random.uniform(self.radius_min, self.radius_max)
                return img.filter(ImageFilter.GaussianBlur(radius=r))

        class Solarization:
            def __init__(self, p: float):
                self.p = p

            def __call__(self, img: Image) -> Image:
                if random.random() > self.p:
                    return img
                return ImageOps.solarize(img)

        def __init__(
            self,
            img_size: int,
            local_crops_number: int,
            local_crops_scale: Tuple[float, float],
            global_crops_scale: Tuple[float, float],
        ):
            flip_and_color_jitter = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.2,
                                hue=0.1,
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )
            normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            # global crop 1
            self.global_transform1 = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        img_size,
                        scale=global_crops_scale,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    flip_and_color_jitter,
                    self.GaussianBlur(1.0),
                    normalize,
                ]
            )
            # global crop 2
            self.global_transform2 = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        img_size,
                        scale=global_crops_scale,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    flip_and_color_jitter,
                    self.GaussianBlur(0.1),
                    self.Solarization(0.2),
                    normalize,
                ]
            )
            # local crop
            self.local_crops_number = local_crops_number
            self.local_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        int(img_size * 3 / 7),
                        scale=local_crops_scale,
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    flip_and_color_jitter,
                    self.GaussianBlur(0.5),
                    normalize,
                ]
            )

        def __call__(self, image: Image) -> Image:
            image = image.convert("RGB")
            crops = [self.global_transform1(image), self.global_transform2(image)]
            for _ in range(self.local_crops_number):
                crops.append(self.local_transform(image))
            return crops

    def __init__(
        self,
        img_size: int,
        local_crops_number: int = 8,
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
    ):
        super().__init__()
        self.fn = self.Augmentation(
            img_size,
            local_crops_number,
            local_crops_scale,
            global_crops_scale,
        )

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("ssl_test")
class SSLTestTransform(Transforms):
    def __init__(self, img_size: int):
        super().__init__()
        self.fn = transforms.Compose(
            [
                Function(lambda img: img.convert("RGB")),
                transforms.Resize(
                    int(round(img_size * 8.0 / 7.0)),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(img_size),
                ToArray(),
                Normalize(),
                transforms.ToTensor(),
            ]
        )

    @property
    def need_batch_process(self) -> bool:
        return False


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


@Transforms.register("to_array")
class ToArray(ATransforms):
    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = lambda **inp: {k: np.array(v) for k, v in inp.items()}


@Transforms.register("to_rgb")
class ToRGB(ATransforms):
    def __init__(self, *, label_alias: Optional[str] = None):
        super().__init__(label_alias=label_alias)
        self.fn = lambda **inp: {k: self._to_rgb(k, v) for k, v in inp.items()}

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


@Transforms.register("normalize")
class Normalize(ATransforms):
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


@Transforms.register("a_bundle")
class ABundle(Compose):
    def __init__(
        self,
        *,
        resize_size: int = 320,
        crop_size: int = 288,
        p: float = 0.5,
        label_alias: Optional[str] = None,
    ):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                RandomCrop(crop_size, label_alias=label_alias),
                HFlip(p, label_alias=label_alias),
                VFlip(p, label_alias=label_alias),
                ShiftScaleRotate(p, cv2.BORDER_CONSTANT, label_alias=label_alias),
                RGBShift(p=p, label_alias=label_alias),
                Solarize(p=p, label_alias=label_alias),
                GaussianBlur(p=p, label_alias=label_alias),
                HueSaturationValue(p=p, label_alias=label_alias),
                RandomBrightnessContrast(p=p, label_alias=label_alias),
                Normalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("a_bundle_test")
class ABundleTest(Compose):
    def __init__(self, *, resize_size: int = 320, label_alias: Optional[str] = None):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                Normalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("style_transfer")
class StyleTransfer(Compose):
    def __init__(
        self,
        *,
        resize_size: int = 512,
        crop_size: int = 256,
        label_alias: Optional[str] = None,
    ):
        super().__init__(
            [
                ToRGB(label_alias=label_alias),
                Resize(resize_size, label_alias=label_alias),
                RandomCrop(crop_size, label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("style_transfer_test")
class StyleTransferTest(Compose):
    def __init__(self, *, resize_size: int = 256, label_alias: Optional[str] = None):
        super().__init__(
            [
                ToRGB(label_alias=label_alias),
                Resize(resize_size, label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


__all__ = [
    "Function",
    "Compose",
    "ToGray",
    "ToTensor",
    "RandomResizedCrop",
    "N1To1",
    "Inverse0To1",
    "InverseN1To1",
    "ForGeneration",
    "ForImagenet",
    "SSLTransform",
    "SSLTestTransform",
    "ToArray",
    "Resize",
    "RandomCrop",
    "ShiftScaleRotate",
    "HFlip",
    "VFlip",
    "Normalize",
    "RGBShift",
    "Solarize",
    "GaussianBlur",
    "HueSaturationValue",
    "RandomBrightnessContrast",
    "AToTensor",
    "ABundle",
    "ABundleTest",
    "StyleTransfer",
    "StyleTransferTest",
]
