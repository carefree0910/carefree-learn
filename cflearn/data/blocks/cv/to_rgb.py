import numpy as np

from PIL import Image
from torch import from_numpy
from torch import Tensor
from typing import Any
from typing import Dict
from cftool.cv import to_rgb
from cftool.types import np_dict_type

from ..common import IRuntimeDataBlock
from ....constants import INPUT_KEY


@IRuntimeDataBlock.register("to_rgb")
class ToRGBBlock(IRuntimeDataBlock):
    def postprocess_item(
        self,
        item: Dict[str, Any],
        for_inference: bool,
    ) -> np_dict_type:
        image = item[INPUT_KEY]
        original_dtype = None
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                original_dtype = "np"
                image = Image.fromarray(image)
            elif isinstance(image, Tensor):
                original_dtype = "torch"
                image = Image.fromarray(image.numpy())
            else:
                raise ValueError(f"unsupported image type: {type(image)}")
        image = to_rgb(image)
        if original_dtype is not None:
            array = np.array(image)
            image = array if original_dtype == "np" else from_numpy(array)
        item[INPUT_KEY] = image
        return item


__all__ = [
    "ToRGBBlock",
]
