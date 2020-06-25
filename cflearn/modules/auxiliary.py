import torch
import torch.nn as nn

from typing import *
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


class EMA(nn.Module):
    def __init__(self,
                 decay: float,
                 named_parameters: Iterable[Tuple[str, nn.Parameter]]):
        super().__init__()
        self._decay, self._named_parameters = decay, list(named_parameters)
        for name, param in self.tgt_params:
            self.register_buffer("tr_" + name, param.data.clone())
            self.register_buffer("ema_" + name, param.data.clone())

    @property
    def tgt_params(self):
        return map(
            lambda pair: (pair[0].replace(".", "_"), pair[1]),
            filter(lambda pair: pair[1].requires_grad, self._named_parameters)
        )

    def forward(self):
        for name, param in self.tgt_params:
            tr_name, ema_name = "tr_" + name, "ema_" + name
            setattr(self, tr_name, param.data.clone())
            ema = (1.0 - self._decay) * param.data + self._decay * getattr(self, ema_name)
            setattr(self, ema_name, ema.clone())

    def train(self, mode: bool = True):
        super().train(mode)
        prefix = "tr_" if mode else "ema_"
        for name, param in self.tgt_params:
            param.data = getattr(self, prefix + name).clone()

    def extra_repr(self) -> str:
        max_str_len = max(len(name) for name, _ in self.tgt_params)
        return "\n".join(
            [f"(0): decay_rate={self._decay}\n(1): Params("] + [
                f"  {name:<{max_str_len}s} - torch.Tensor({list(param.shape)})"
                for name, param in self.tgt_params
            ] + [")"]
        )
