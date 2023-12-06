from typing import List
from typing import Union

from ..common import get_wh
from ..common import IAlbumentationsBlock

try:
    from albumentations import CenterCrop
    from albumentations import RandomCrop
except:
    CenterCrop = RandomCrop = None


@IAlbumentationsBlock.register("center_crop")
class CenterCropBlock(IAlbumentationsBlock):
    size: Union[int, List[int]]
    always_apply: bool
    p: float

    def __init__(
        self,
        size: Union[int, List[int]] = 512,
        *,
        always_apply: bool = False,
        p: float = 1.0
    ) -> None:
        super().__init__(size=size, always_apply=always_apply, p=p)

    @property
    def fields(self) -> List[str]:
        return ["size", "always_apply", "p"]

    def init_fn(self, for_inference: bool) -> CenterCrop:
        w, h = get_wh(self.size)
        return CenterCrop(h, w, self.always_apply, self.p)


@IAlbumentationsBlock.register("random_crop")
class RandomCropBlock(IAlbumentationsBlock):
    size: Union[int, List[int]]
    always_apply: bool
    p: float

    def __init__(
        self,
        size: Union[int, List[int]] = 512,
        *,
        always_apply: bool = False,
        p: float = 1.0
    ) -> None:
        super().__init__(size=size, always_apply=always_apply, p=p)

    @property
    def fields(self) -> List[str]:
        return ["size", "always_apply", "p"]

    def init_fn(self, for_inference: bool) -> Union[CenterCrop, RandomCrop]:
        w, h = get_wh(self.size)
        if for_inference:
            return CenterCrop(h, w, self.always_apply, self.p)
        return RandomCrop(h, w, self.always_apply, self.p)


__all__ = [
    "CenterCropBlock",
    "RandomCropBlock",
]
