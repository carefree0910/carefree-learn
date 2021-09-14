import numpy as np

from typing import Dict
from typing import Callable

from ....misc.toolkit import contrast_noise
from ....misc.toolkit import fractal_noise_2d
from ....misc.toolkit import min_max_normalize

noise_fn_type = Callable[[int, int], np.ndarray]
noises: Dict[str, noise_fn_type] = {}


def register_noise(name: str) -> Callable[[noise_fn_type], noise_fn_type]:
    def _core(fn: noise_fn_type) -> noise_fn_type:
        noises[name] = fn
        return fn

    return _core


@register_noise("fractal")
def fractal_noise(w: int, h: int) -> np.ndarray:
    if w > 1024 or h > 1024:
        side, octaves = 2048, 7
    elif w > 512 or h > 512:
        side, octaves = 1024, 6
    elif w > 256 or h > 256:
        side, octaves = 512, 5
    else:
        side, octaves = 256, 4

    r = min_max_normalize(fractal_noise_2d((side, side), (32, 32), octaves))
    g = min_max_normalize(fractal_noise_2d((side, side), (32, 32), octaves))
    b = min_max_normalize(fractal_noise_2d((side, side), (32, 32), octaves))
    stack = np.dstack((contrast_noise(r), contrast_noise(g), contrast_noise(b)))
    return (255.0 * stack).astype(np.uint8)


__all__ = ["noises"]
