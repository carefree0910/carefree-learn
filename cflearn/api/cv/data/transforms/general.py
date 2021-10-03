import numpy as np

from typing import Any

from .....data import Transforms
from .....types import np_dict_type


class NoBatchTransforms(Transforms):
    @property
    def need_batch_process(self) -> bool:
        return False


@Transforms.register("to_array")
class ToArray(Transforms):
    @staticmethod
    def fn(**inp: Any) -> np_dict_type:
        return {k: np.asarray(v).astype(np.float32) / 255.0 for k, v in inp.items()}

    @property
    def need_numpy(self) -> bool:
        return False

    @property
    def need_batch_process(self) -> bool:
        return True


@Transforms.register("to_rgb")
class ToRGB(NoBatchTransforms):
    @staticmethod
    def fn(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            img = img[..., None]
        if img.shape[2] == 3:
            return img
        if img.shape[2] == 4:
            return img[..., :3] * img[..., 3:]
        if img.shape[2] == 1:
            return img.repeat(3, axis=2)
        raise ValueError(f"invalid shape occurred ({img.shape})")

    @property
    def need_numpy(self) -> bool:
        return True


__all__ = [
    "ToRGB",
    "ToArray",
    "NoBatchTransforms",
]
