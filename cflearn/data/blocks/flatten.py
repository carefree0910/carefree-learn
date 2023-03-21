from cftool.types import np_dict_type

from ...schema import IRuntimeDataBlock
from ...constants import INPUT_KEY


@IRuntimeDataBlock.register("flatten")
class FlattenBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: np_dict_type) -> np_dict_type:
        item[INPUT_KEY] = item[INPUT_KEY].ravel()
        return item


__all__ = ["FlattenBlock"]
