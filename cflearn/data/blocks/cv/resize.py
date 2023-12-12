from typing import List
from typing import Union

from ..common import get_wh
from ..common import IAlbumentationsBlock


try:
    from cv2 import INTER_LINEAR
    from albumentations import Resize
except:
    INTER_LINEAR = Resize = None


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


__all__ = [
    "ResizeBlock",
]
