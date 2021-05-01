import math


def f_map_dim(img_size: int, num_layer: int) -> int:
    return int(round(img_size / 2 ** num_layer))


def num_downsample(img_size: int, *, min_size: int = 2) -> int:
    return max(2, int(round(math.log2(img_size / min_size))))
