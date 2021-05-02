import math


def f_map_dim(img_size: int, num_layer: int) -> int:
    return int(round(img_size / 2 ** num_layer))


def auto_num_downsample(
    img_size: int,
    min_size: int = 4,
    target_downsample: int = 4,
) -> int:
    max_downsample = int(round(math.log2(img_size / min_size)))
    return max(2, min(target_downsample, max_downsample))
