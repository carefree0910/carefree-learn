from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from cftool.types import np_dict_type

from ...schema import DataBundle
from ...schema import IDataBlock
from ...constants import INPUT_KEY

try:
    from albumentations import BasicTransform
except:
    BasicTransform = None


def get_wh(size: Union[int, List[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return tuple(size)  # type: ignore


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


class IAlbumentationsBlock(IRuntimeDataBlock):
    fns: Dict[bool, BasicTransform]

    @abstractmethod
    def init_fn(self, for_inference: bool) -> BasicTransform:
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if BasicTransform is None:
            name = self.__class__.__name__
            raise ValueError(f"`albumentations` is required for `{name}`")
        super().__init__(*args, **kwargs)
        self.fns = {}

    def postprocess_item(self, item: np_dict_type, for_inference: bool) -> np_dict_type:
        fn = self.fns.get(for_inference)
        if fn is None:
            fn = self.fns[for_inference] = self.init_fn(for_inference)
        item[INPUT_KEY] = fn(image=item[INPUT_KEY])["image"]
        return item


@IRuntimeDataBlock.register("flatten")
class FlattenBlock(IRuntimeDataBlock):
    def postprocess_item(self, item: np_dict_type, for_inference: bool) -> np_dict_type:
        item[INPUT_KEY] = item[INPUT_KEY].ravel()
        return item


__all__ = [
    "IRuntimeDataBlock",
    "FlattenBlock",
]
