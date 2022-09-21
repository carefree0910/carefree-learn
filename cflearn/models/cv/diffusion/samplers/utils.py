import torch

from torch import Tensor


def append_dims(net: Tensor, ndim: int) -> Tensor:
    diff = ndim - net.ndim
    return net[(...,) + (None,) * diff]


def append_zero(net: Tensor) -> Tensor:
    return torch.cat([net, net.new_zeros([1])])
