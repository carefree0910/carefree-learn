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


class Activations:
    """
    Wrapper class for pytorch activations
    * when pytorch implemented corresponding activation, it will be returned
    * otherwise, custom implementation will be returned

    Parameters
    ----------
    configs : {None, dict}, configuration for the activation

    Examples
    --------
    >>> act = Activations()
    >>> print(type(act.ReLU))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.module("ReLU")))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.Tanh))  # <class 'nn.modules.activation.Tanh'>
    >>> print(type(act.one_hot))  # <class '__main__.Activations.one_hot.<locals>.OneHot'>

    """

    def __init__(self,
                 configs: Dict[str, Any] = None):
        if configs is None:
            configs = {}
        self.configs = configs

    def __getattr__(self, item):
        try:
            return getattr(nn, item)(**self.configs.setdefault(item, {}))
        except AttributeError:
            raise NotImplementedError(
                f"neither pytorch nor custom Activations implemented activation '{item}'")

    def module(self,
               name: str) -> nn.Module:
        if name is None:
            return nn.Identity()
        return getattr(self, name)

    # publications

    @property
    def mish(self):

        class Mish(nn.Module):
            def forward(self, x):
                return x * (torch.tanh(nn.functional.softplus(x)))

        return Mish()

    # custom

    @property
    def sign(self):

        class Sign(nn.Module):
            def forward(self, x):
                return torch.sign(x)

        return Sign()

    @property
    def one_hot(self):

        class OneHot(nn.Module):
            def forward(self, x):
                return x * (x == torch.max(x, dim=1, keepdim=True)[0]).to(torch.float32)

        return OneHot()

    @property
    def multiplied_tanh(self):

        class MultipliedTanh(nn.Tanh):
            def __init__(self, ratio, trainable=True):
                super().__init__()
                ratio = torch.tensor([ratio], dtype=torch.float32)
                self.ratio = ratio if not trainable else nn.Parameter(ratio)

            def forward(self, x):
                x = x * self.ratio
                return super().forward(x)

        return MultipliedTanh(**self.configs.setdefault("multiplied_tanh", {}))

    @property
    def multiplied_softmax(self):

        class MultipliedSoftmax(nn.Softmax):
            def __init__(self, ratio, dim=1, trainable=True):
                super().__init__(dim)
                ratio = torch.tensor([ratio], dtype=torch.float32)
                self.ratio = ratio if not trainable else nn.Parameter(ratio)

            def forward(self, x):
                x = x * self.ratio
                return super().forward(x)

        return MultipliedSoftmax(**self.configs.setdefault("multiplied_softmax", {}))
