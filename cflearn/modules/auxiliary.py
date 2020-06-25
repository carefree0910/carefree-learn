import torch
import torch.nn as nn

from functools import partial
from torch.nn.functional import softplus


class BN(nn.BatchNorm1d):
    def forward(self, net):
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


class Dropout(nn.Module):
    def __init__(self, dropout: float):
        if dropout < 0. or dropout > 1.:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {dropout}")
        super().__init__()
        self._mask_cache = None
        self._keep_prob = 1. - dropout

    def forward(self, net, *, reuse: bool = False):
        if not self.training:
            return net
        if reuse:
            mask = self._mask_cache
        else:
            self._mask_cache = mask = torch.bernoulli(
                net.new(*net.shape).fill_(self._keep_prob)) / self._keep_prob
        net = net * mask
        del mask
        return net

    def extra_repr(self) -> str:
        return f"keep={self._keep_prob}"
