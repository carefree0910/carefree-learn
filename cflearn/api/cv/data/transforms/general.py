import numpy as np

from typing import Any
from typing import Dict
from torchvision.transforms import transforms

from .....data import Transforms


class NoBatchTransforms(Transforms):
    @property
    def need_batch_process(self) -> bool:
        return False


class BatchWrapper(Transforms):
    def __init__(self, transform: Transforms, input_alias: str):
        super().__init__()
        self.transform = transform
        self.input_alias = input_alias

    def fn(self, inp: Dict[str, Any]) -> Dict[str, Any]:
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
    def __init__(self, need_batch_process: bool = False):
        super().__init__()
        self._batch = need_batch_process

    @staticmethod
    def to_array(v: Any) -> np.ndarray:
        return np.asarray(v).astype(np.float32) / 255.0

    def fn(self, inp: Any) -> Any:
        if not self._batch:
            return self.to_array(inp)
        return {k: self.to_array(v) for k, v in inp.items()}

    @property
    def need_numpy(self) -> bool:
        return False

    @property
    def need_batch_process(self) -> bool:
        return self._batch


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
    def __init__(self):  # type: ignore
        super().__init__()
        self.pil_fn = transforms.Grayscale()

    @staticmethod
    def np_fn(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img[..., None]
        if img.shape[2] == 3:
            return img.mean(axis=2, keepdims=True)
        if img.shape[2] == 4:
            return img[..., :3].mean(axis=2, keepdims=True) * img[..., 3:]
        if img.shape[2] == 1:
            return img
        raise ValueError(f"invalid shape occurred ({img.shape})")

    def fn(self, inp: Any) -> Any:
        if isinstance(inp, np.ndarray):
            return self.np_fn(inp)
        return self.pil_fn(inp)

    @property
    def need_numpy(self) -> bool:
        return False


__all__ = [
    "ToRGB",
    "ToGray",
    "ToArray",
    "BatchWrapper",
    "NoBatchTransforms",
]