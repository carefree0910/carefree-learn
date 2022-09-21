from torch import Tensor


def append_dims(net: Tensor, ndim: int) -> Tensor:
    diff = ndim - net.ndim
    return net[(...,) + (None,) * diff]
