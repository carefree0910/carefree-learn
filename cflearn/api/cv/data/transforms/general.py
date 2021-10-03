import numpy as np

from typing import Any
from typing import Dict

from .....data import Transforms
from .....types import np_dict_type


class NoBatchTransforms(Transforms):
    @property
    def need_batch_process(self) -> bool:
        return False


class BatchWrapper(Transforms):
    def __init__(self, transform: Transforms, input_alias: str):
        super().__init__()
        self.transform = transform
        self.input_alias = input_alias

    def fn(self, **inp: Any) -> Dict[str, Any]:
        return {
            key: value if key != self.input_alias else self.transform(value)
            for key, value in inp.items()
        }

    @property
    def need_numpy(self) -> bool:
        return self.transform.need_numpy

    @property
    def need_batch_process(self) -> bool:
        return True


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


@Transforms.register("to_gray")
class ToGray(NoBatchTransforms):
    @staticmethod
    def fn(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img[..., None]
        if img.shape[2] == 3:
            return img.mean(axis=2, keepdims=True)
        if img.shape[2] == 4:
            return img[..., :3].mean(axis=2, keepdims=True) * img[..., 3:]
        if img.shape[2] == 1:
            return img
        raise ValueError(f"invalid shape occurred ({img.shape})")

    @property
    def need_numpy(self) -> bool:
        return True


__all__ = [
    "ToRGB",
    "ToGray",
    "ToArray",
    "BatchWrapper",
    "NoBatchTransforms",
]
