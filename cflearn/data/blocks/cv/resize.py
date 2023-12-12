from typing import List
from typing import Union
from typing import Optional
from cftool.cv import restrict_wh
from cftool.cv import get_suitable_size
from cftool.types import np_dict_type

from ..common import get_wh
from ..common import IRuntimeDataBlock
from ..common import IAlbumentationsBlock
from ....constants import INPUT_KEY


try:
    from cv2 import INTER_LINEAR
    from albumentations import Resize
    from albumentations.augmentations.geometric.functional import resize
except:
    INTER_LINEAR = Resize = resize = None


@IAlbumentationsBlock.register("resize")
class ResizeBlock(IAlbumentationsBlock):
    size: Union[int, List[int]]
    interpolation: int

    def __init__(
        self,
        size: Union[int, List[int]] = 512,
        interpolation: int = INTER_LINEAR,
    ) -> None:
        super().__init__(size=size, interpolation=interpolation)

    @property
    def fields(self) -> List[str]:
        return ["size", "interpolation"]

    def init_fn(self, for_inference: bool) -> Resize:
        w, h = get_wh(self.size)
        return Resize(h, w, self.interpolation)


@IRuntimeDataBlock.register("anchored_resize")
class AnchoredResizeBlock(IRuntimeDataBlock):
    anchor: Optional[int]
    max_wh: Optional[int]
    interpolation: int

    def __init__(
        self,
        *,
        anchor: Optional[int] = None,
        max_wh: Optional[int] = None,
        interpolation: int = INTER_LINEAR,
    ) -> None:
        if resize is None:
            name = self.__class__.__name__
            raise ValueError(f"`albumentations` is required for `{name}`")
        super().__init__(anchor=anchor, max_wh=max_wh, interpolation=interpolation)

    @property
    def fields(self) -> List[str]:
        return ["anchor", "max_wh", "interpolation"]

    def postprocess_item(self, item: np_dict_type, for_inference: bool) -> np_dict_type:
        image = item[INPUT_KEY]
        h, w = image.shape[:2]
        if self.max_wh is not None:
            h, w = restrict_wh(h, w, self.max_wh)
        if self.anchor is not None:
            h, w = map(get_suitable_size, [h, w], [self.anchor, self.anchor])
        image = resize(image, h, w, self.interpolation)
        item[INPUT_KEY] = image
        return item


__all__ = [
    "ResizeBlock",
    "AnchoredResizeBlock",
]
