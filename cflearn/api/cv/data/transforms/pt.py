from typing import Any
from typing import Tuple
from typing import Union
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode

from .general import NoBatchTransforms
from .....data import Transforms


@Transforms.register("to_gray")
class ToGray(NoBatchTransforms):
    fn = transforms.Grayscale()


@Transforms.register("to_tensor")
class ToTensor(NoBatchTransforms):
    fn = transforms.ToTensor()


@Transforms.register("resize")
class Resize(NoBatchTransforms):
    def __init__(
        self,
        size: Union[int, tuple],
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        super().__init__()
        if isinstance(size, int):
            size = size, size
        self.fn = transforms.Resize(size, interpolation)


@Transforms.register("random_erase")
class RandomErase(NoBatchTransforms):
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


@Transforms.register("random_resized_crop")
class RandomResizedCrop(NoBatchTransforms):
    def __init__(self, *, size: int = 224):
        super().__init__()
        self.fn = transforms.RandomResizedCrop(size)


@Transforms.register("color_jitter")
class ColorJitter(NoBatchTransforms):
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


@Transforms.register("normalize")
class Normalize(NoBatchTransforms):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.fn = transforms.Normalize(mean, std)


@Transforms.register("-1~1")
class N1To1(NoBatchTransforms):
    fn = transforms.Lambda(lambda t: t * 2.0 - 1.0)


@Transforms.register("inverse_0~1")
class Inverse0To1(NoBatchTransforms):
    fn = transforms.Lambda(lambda t: 1.0 - t)


@Transforms.register("inverse_-1~1")
class InverseN1To1(NoBatchTransforms):
    fn = transforms.Lambda(lambda t: -t)


__all__ = [
    "ToGray",
    "ToTensor",
    "Resize",
    "RandomErase",
    "RandomResizedCrop",
    "ColorJitter",
    "Normalize",
    "N1To1",
    "Inverse0To1",
    "InverseN1To1",
]
