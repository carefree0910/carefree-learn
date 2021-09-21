import cv2
import random

import numpy as np

from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from torch import Tensor
from typing import List
from typing import Tuple
from typing import Optional
from skimage.transform import resize
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode

from .A import *
from .pt import *
from .protocol import Compose
from .protocol import Transforms
from .....misc.toolkit import min_max_normalize
from .....misc.toolkit import imagenet_normalize


@Transforms.register("for_generation")
class TransformForGeneration(Compose):
    def __init__(
        self,
        img_size: Optional[int] = None,
        *,
        inverse: bool = False,
        to_gray: bool = False,
        to_rgb: bool = False,
    ):
        transform_list: List[Transforms] = []
        if img_size is not None:
            transform_list.append(Resize(img_size))
        if to_rgb:
            if to_gray:
                msg = "should not use `to_rgb` and `to_gray` at the same time"
                raise ValueError(msg)
            transform_list.append(ToRGB())
        elif to_gray:
            transform_list.append(AToGray())
        transform_list.extend([ToTensor(), N1To1()])
        if inverse:
            transform_list.append(InverseN1To1())
        super().__init__(transform_list)


@Transforms.register("for_imagenet")
class TransformForImagenet(Compose):
    def __init__(self, img_size: int = 224):  # type: ignore
        super().__init__([Resize(img_size), ANormalize(), ToTensor()])


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
                    ColorJitter(p=0.8),
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
        self.img_size = img_size
        self.larger_size = int(round(img_size * 8.0 / 7.0))

    def fn(self, img: Image.Image) -> Tensor:
        img = img.convert("RGB")
        img.thumbnail((self.larger_size, self.larger_size), Image.ANTIALIAS)
        img_arr = np.array(img)
        resized_img = resize(img_arr, (self.img_size, self.img_size), mode="constant")
        resized_img = resized_img.astype(np.float32)
        img_arr = min_max_normalize(resized_img)
        img_arr = imagenet_normalize(img_arr)
        return img_arr.transpose([2, 0, 1])

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("a_bundle")
class ABundleTransform(Compose):
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
                ANormalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("a_bundle_test")
class ABundleTestTransform(Compose):
    def __init__(self, *, resize_size: int = 320, label_alias: Optional[str] = None):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                ANormalize(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("style_transfer")
class StyleTransferTransform(Compose):
    def __init__(
        self,
        *,
        resize_size: int = 512,
        crop_size: int = 256,
        label_alias: Optional[str] = None,
    ):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                RandomCrop(crop_size, label_alias=label_alias),
                ToRGB(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("style_transfer_test")
class StyleTransferTestTransform(Compose):
    def __init__(self, *, resize_size: int = 256, label_alias: Optional[str] = None):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                ToRGB(label_alias=label_alias),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("munit")
class MUNITTransform(Compose):
    def __init__(
        self,
        *,
        p: float = 0.5,
        resize_size: int = 256,
        label_alias: Optional[str] = None,
    ):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                HFlip(p, label_alias=label_alias),
                ToRGB(label_alias=label_alias),
                ANormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("munit_test")
class MUNITTestTransform(Compose):
    def __init__(self, *, resize_size: int = 256, label_alias: Optional[str] = None):
        super().__init__(
            [
                Resize(resize_size, label_alias=label_alias),
                ToRGB(label_alias=label_alias),
                ANormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                AToTensor(label_alias=label_alias),
            ]
        )


@Transforms.register("clf")
class ClassificationTransform(Compose):
    def __init__(
        self,
        *,
        p: float = 0.5,
        resize_size: int = 512,
        label_alias: Optional[str] = None,
    ):
        if label_alias is not None:
            raise ValueError("`label_alias` should not be provided in `Classification`")
        super().__init__(
            [
                Resize(int(resize_size * 1.2)),
                ToRGB(),
                RandomCrop(resize_size),
                HFlip(p),
                AToTensor(),
                ColorJitter(p=min(1.0, p * 1.6)),
                RandomErase(p=p),
                Normalize(),
            ]
        )


@Transforms.register("clf_test")
class ClassificationTestTransform(Compose):
    def __init__(self, *, resize_size: int = 512, label_alias: Optional[str] = None):
        if label_alias is not None:
            raise ValueError("`label_alias` should not be provided in `Classification`")
        super().__init__([Resize(resize_size), ToRGB(), ANormalize(), AToTensor()])


__all__ = [
    "TransformForGeneration",
    "TransformForImagenet",
    "SSLTransform",
    "SSLTestTransform",
    "ABundleTransform",
    "ABundleTestTransform",
    "StyleTransferTransform",
    "StyleTransferTestTransform",
    "MUNITTransform",
    "MUNITTestTransform",
    "ClassificationTransform",
    "ClassificationTestTransform",
]
