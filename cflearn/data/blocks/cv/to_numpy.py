import numpy as np

from PIL import Image
from torch import Tensor
from typing import Any
from typing import Dict
from cftool.types import np_dict_type

from ....schema import IRuntimeDataBlock
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY


@IRuntimeDataBlock.register("to_numpy")
class ToNumpyBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: Dict[str, Any]) -> np_dict_type:
        image = item[INPUT_KEY]
        labels = item[LABEL_KEY]
        if isinstance(image, Tensor):
            image = image.numpy()
        elif isinstance(image, Image.Image):
            if image.mode == "P":
                image = image.convert("RGBA")
            image = np.array(image)
            if len(image.shape) == 2:
                image = image[..., None]
        item[INPUT_KEY] = image
        if isinstance(labels, (int, float)):
            item[LABEL_KEY] = np.array([labels])
        return item


__all__ = ["ToNumpyBlock"]
