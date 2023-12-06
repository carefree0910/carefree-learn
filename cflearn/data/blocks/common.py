from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import List
from cftool.types import np_dict_type

from ...schema import DataBundle
from ...schema import IDataBlock
from ...constants import INPUT_KEY


class IRuntimeDataBlock(IDataBlock, metaclass=ABCMeta):
    """
    Runtime blocks will store no information, and will only process the batches
    at runtime. When dealing with CV/NLP datasets, we'll often use this kind of blocks.
    """

    @property
    def fields(self) -> List[str]:
        return []

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        return bundle

    @abstractmethod
    def postprocess_item(self, item: Any, for_inference: bool) -> Any:
        """changes can happen inplace"""


@IRuntimeDataBlock.register("flatten")
class FlattenBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: np_dict_type, for_inference: bool) -> np_dict_type:
        item[INPUT_KEY] = item[INPUT_KEY].ravel()
        return item


__all__ = [
    "IRuntimeDataBlock",
    "FlattenBlock",
]
