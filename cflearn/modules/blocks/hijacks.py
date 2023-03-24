import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Optional

from .hooks import ConvHook
from .hooks import LinearHook


class HijackLinear(nn.Linear):
    def __init__(self, *args: Any, hook: Optional[LinearHook] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hook = hook

    def forward(self, net: Tensor) -> Tensor:
        net = super().forward(net)
        if self.hook is not None:
            net = self.hook.callback(net)
        return net


class HijackConv1d(nn.Conv1d):
    def __init__(self, *args: Any, hook: Optional[ConvHook] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hook = hook

    def forward(self, net: Tensor) -> Tensor:
        net = super().forward(net)
        if self.hook is not None:
            net = self.hook.callback(net)
        return net


class HijackConv2d(nn.Conv2d):
    def __init__(self, *args: Any, hook: Optional[ConvHook] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hook = hook

    def forward(self, net: Tensor) -> Tensor:
        net = super().forward(net)
        if self.hook is not None:
            net = self.hook.callback(net)
        return net


class HijackConv3d(nn.Conv3d):
    def __init__(self, *args: Any, hook: Optional[ConvHook] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hook = hook

    def forward(self, net: Tensor) -> Tensor:
        net = super().forward(net)
        if self.hook is not None:
            net = self.hook.callback(net)
        return net
