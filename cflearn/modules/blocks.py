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
