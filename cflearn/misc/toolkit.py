import math
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import *

from cftool.misc import *


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Initializer(LoggingMixin):
    """
    Initializer for neural network weights

    Examples
    --------
    >>> initializer = Initializer({})
    >>> linear = nn.Linear(10, 10)
    >>> initializer.xavier(linear.weight)

    """

    defined_initialization = {"xavier", "normal", "truncated_normal"}
    custom_initializer = {}

    def __init__(self, config):
        self.config = config
        self._verbose_level = config.setdefault("verbose_level", 2)

    def initialize(self, param: nn.Parameter, method: str):
        custom_initializer = self.custom_initializer.get(method)
        if custom_initializer is None:
            return getattr(self, method)(param)
        return custom_initializer(self, param)

    @classmethod
    def add_initializer(cls, f, name):
        if name in cls.defined_initialization:
            print(f"{cls.warning_prefix}'{name}' initializer is already defined")
            return
        cls.defined_initialization.add(name)
        cls.custom_initializer[name] = f

    @staticmethod
    def xavier(param: nn.Parameter):
        nn.init.xavier_uniform_(param.data)

    def normal(self, param: nn.Parameter):
        mean = self.config.setdefault("mean", 0.)
        std = self.config.setdefault("std", 1.)
        with torch.no_grad():
            param.data.normal_(mean, std)

    def truncated_normal(self, param: nn.Parameter):
        span = self.config.setdefault("span", 2.)
        mean = self.config.setdefault("mean", 0.)
        std = self.config.setdefault("std", 1.)
        tol = self.config.setdefault("tol", 0.)
        epoch = self.config.setdefault("epoch", 20)
        n_elem = param.numel()
        weight_base = param.new_empty(n_elem).normal_()
        get_invalid = lambda w: (w > span) | (w < -span)
        invalid = get_invalid(weight_base)
        success = False
        for _ in range(epoch):
            n_invalid = int(invalid.sum())
            if n_invalid / n_elem <= tol:
                success = True
                break
            with torch.no_grad():
                weight_base[invalid] = param.new_empty(n_invalid).normal_()
                invalid = get_invalid(weight_base)
        if not success:
            self.log_msg(
                f"invalid ratio for truncated normal : {invalid.to(torch.float32).mean():8.6f}, "
                f"it might cause by too little epoch ({epoch}) or too small tolerance ({tol})",
                self.warning_prefix, 2, logging.WARNING
            )
        with torch.no_grad():
            param.data.copy_(weight_base.reshape(param.shape))
            param.data.mul_(std).add_(mean)
