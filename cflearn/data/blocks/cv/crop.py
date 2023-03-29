import torch

import numpy as np

from typing import List
from typing import Tuple
from typing import Union
from torchvision import transforms
from cftool.types import np_dict_type

from ....schema import IRuntimeDataBlock
from ....constants import INPUT_KEY


@IRuntimeDataBlock.register("center_crop")
class CenterCropBlock(IRuntimeDataBlock):
    size: Union[int, List[int]]

    def __init__(self, size: Union[int, List[int]] = 512) -> None:
        super().__init__(size=size)

    @property
    def fields(self) -> List[str]:
        return ["size"]

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        tensor = torch.from_numpy(item[INPUT_KEY])
        tensor = transforms.CenterCrop(self.size)(tensor)
        item[INPUT_KEY] = tensor.numpy()
        return item


@IRuntimeDataBlock.register("random_crop")
class RandomCropBlock(IRuntimeDataBlock):
    size: Union[int, List[int]]

    def __init__(self, size: Union[int, List[int]] = 512) -> None:
        super().__init__(size=size)

    @property
    def fields(self) -> List[str]:
        return ["size"]

    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        tensor = torch.from_numpy(item[INPUT_KEY])
        tensor = transforms.RandomCrop(self.size)(tensor)
        item[INPUT_KEY] = tensor.numpy()
        return item


__all__ = [
    "CenterCropBlock",
    "RandomCropBlock",
]
