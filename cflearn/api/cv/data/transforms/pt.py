from typing import Any
from typing import Tuple
from torchvision.transforms import transforms

from .....misc.internal_.data import Transforms


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


@Transforms.register("random_erase")
class RandomErase(Transforms):
    def __init__(
        self,
        *,
        p: float,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Any = 0,
        inplace: bool = False,
    ):
        super().__init__()
        self.fn = transforms.RandomErasing(p, scale, ratio, value, inplace)

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


@Transforms.register("color_jitter")
class ColorJitter(Transforms):
    def __init__(
        self,
        *,
        p: float,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        super().__init__()
        self.fn = transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                )
            ],
            p=p,
        )

    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("normalize")
class Normalize(Transforms):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.fn = transforms.Normalize(mean, std)

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


__all__ = [
    "ToGray",
    "ToTensor",
    "RandomErase",
    "RandomResizedCrop",
    "ColorJitter",
    "Normalize",
    "N1To1",
    "Inverse0To1",
    "InverseN1To1",
]
