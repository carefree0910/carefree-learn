import numpy as np

from typing import Any
from typing import Dict
from typing import List
from cftool.misc import PureFromInfoMixin
from cftool.types import np_dict_type

from ....schema import IRuntimeDataBlock
from ....schema import DataProcessorConfig
from ....constants import INPUT_KEY


@IRuntimeDataBlock.register("static_normalize")
class StaticNormalizeBlock(PureFromInfoMixin, IRuntimeDataBlock):
    division: float

    def build(self, config: DataProcessorConfig) -> None:
        configs = (config.block_configs or {}).setdefault("static_normalize", {})
        self.division = configs.setdefault("division", 255.0)

    def to_info(self) -> Dict[str, Any]:
        return dict(division=self.division)

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        item[INPUT_KEY] = item[INPUT_KEY].astype(np.float64) / self.division
        return item


@IRuntimeDataBlock.register("affine_normalize")
class AffineNormalizeBlock(PureFromInfoMixin, IRuntimeDataBlock):
    center: float
    scale: float

    def build(self, config: DataProcessorConfig) -> None:
        configs = (config.block_configs or {}).setdefault("affine_normalize", {})
        self.center = configs.setdefault("center", 0.5)
        self.scale = configs.setdefault("scale", 0.5)

    def to_info(self) -> Dict[str, Any]:
        return dict(center=self.center, scale=self.scale)

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        inp = item[INPUT_KEY].astype(np.float64)
        inp = (inp - self.center) / self.scale
        item[INPUT_KEY] = inp
        return item


@IRuntimeDataBlock.register("imagenet_normalize")
class ImagenetNormalizeBlock(PureFromInfoMixin, IRuntimeDataBlock):
    mean: List[float]
    std: List[float]

    def build(self, config: DataProcessorConfig) -> None:
        configs = (config.block_configs or {}).setdefault("imagenet_normalize", {})
        self.mean = configs.setdefault("mean", [0.485, 0.456, 0.406])
        self.std = configs.setdefault("std", [0.229, 0.224, 0.225])

    def to_info(self) -> Dict[str, Any]:
        return dict(mean=self.mean, std=self.std)

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
