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
        return {k: np.array(v).astype(np.float32) / 255.0 for k, v in inp.items()}

    @property
    def need_numpy(self) -> bool:
        return False

    @property
    def need_batch_process(self) -> bool:
        return True


__all__ = [
    "ToArray",
    "NoBatchTransforms",
]
