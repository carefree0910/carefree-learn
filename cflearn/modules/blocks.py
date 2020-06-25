import torch

import numpy as np
import torch.nn as nn

from typing import Union

from .auxiliary import *
from ..misc.toolkit import *


class Linear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 bias: bool = True,
                 pruner: Pruner = None,
                 init_method: Union[str, None] = "xavier",
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        self.config, self.pruner = kwargs, pruner
        self._use_bias, self._init_method = bias, init_method

    def forward(self, net):
        weight = self.linear.weight if self.pruner is None else self.pruner(self.linear.weight)
        return nn.functional.linear(net, weight, self.linear.bias)

    def reset_parameters(self):
        if self._init_method is None:
            return
        if self._init_method not in Initializer.defined_initialization:
            self.linear.reset_parameters()
        else:
            initializer = Initializer(self.config.setdefault("initialize_config", {}))
            initializer.initialize(self.linear.weight, self._init_method)
            bias_fill = self.config.setdefault("bias_fill", 0.)
            if self._use_bias:
                with torch.no_grad():
                    self.linear.bias.data.fill_(bias_fill)


class Mapping(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 *,
                 bias: bool = True,
                 pruner_config: dict = None,
                 dropout: bool = True,
                 batch_norm: bool = True,
                 activation: str = "ReLU",
                 init_method: str = "xavier",
                 **kwargs):
        super().__init__()
        self.config = kwargs
        pruner = None if pruner_config is None else Pruner(pruner_config)
        self.linear = Linear(in_dim, out_dim, bias=bias, pruner=pruner, init_method=init_method, **kwargs)
        self.bn = None if not batch_norm else BN(out_dim)
        if activation is None:
            self.activation = None
        else:
            activations_ins = Activations(self.config.setdefault("activation_config", None))
            self.activation = activations_ins.module(activation)
        self.dropout = None if not dropout else Dropout(self.config.setdefault("drop_prob", 0.5))

    def forward(self, net, *, reuse: bool = False):
        net = self.linear(net)
        if self.bn is not None:
            net = self.bn(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout is not None:
            net = self.dropout(net, reuse=reuse)
        return net

    def reset_parameters(self):
        self.linear.reset_parameters()
