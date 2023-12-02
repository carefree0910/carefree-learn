import numpy as np

from typing import List
from typing import Tuple
from cftool.types import np_dict_type

from ....schema import IRuntimeDataBlock
from ....constants import INPUT_KEY


@IRuntimeDataBlock.register("static_normalize")
class StaticNormalizeBlock(IRuntimeDataBlock):
    division: float

    def __init__(self, division: float = 255.0) -> None:
        super().__init__(division=division)

    @property
    def fields(self) -> List[str]:
        return ["division"]

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        item[INPUT_KEY] = item[INPUT_KEY].astype(np.float64) / self.division
        return item


@IRuntimeDataBlock.register("affine_normalize")
class AffineNormalizeBlock(IRuntimeDataBlock):
    center: float
    scale: float

    def __init__(self, center: float = 0.5, scale: float = 0.5) -> None:
        super().__init__(center=center, scale=scale)

    @property
    def fields(self) -> List[str]:
        return ["center", "scale"]

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        inp = item[INPUT_KEY].astype(np.float64)
        inp = (inp - self.center) / self.scale
        item[INPUT_KEY] = inp
        return item


@IRuntimeDataBlock.register("imagenet_normalize")
class ImagenetNormalizeBlock(IRuntimeDataBlock):
    mean: List[float]
    std: List[float]

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(mean=list(mean), std=list(std))

    @property
    def fields(self) -> List[str]:
        return ["mean", "std"]

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        inp = item[INPUT_KEY].astype(np.float64)
        inp = (inp - self.mean) / self.std
        item[INPUT_KEY] = inp
        return item


__all__ = [
    "StaticNormalizeBlock",
    "AffineNormalizeBlock",
    "ImagenetNormalizeBlock",
]
