import numpy as np

from cftool.types import np_dict_type

from ....schema import IRuntimeDataBlock
from ....constants import INPUT_KEY


@IRuntimeDataBlock.register("hwc_to_chw")
class HWCToCHWBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        inp = item[INPUT_KEY].transpose([2, 0, 1])
        inp = np.ascontiguousarray(inp)
        item[INPUT_KEY] = inp
        return item


__all__ = ["HWCToCHWBlock"]
