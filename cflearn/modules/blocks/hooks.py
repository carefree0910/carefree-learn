from abc import abstractmethod
from abc import ABCMeta
from torch import nn
from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from torch.nn import Module


TQKV = Tuple[Tensor, Tensor, Tensor]


class IHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, inp: Any, out: Any) -> Any:
        pass


class IBasicHook(IHook):
    @abstractmethod
    def callback(self, inp: Tensor, out: Tensor) -> Tensor:
        pass


class IAttentionHook(IHook):
    @abstractmethod
    def callback(self, qkv_inp: TQKV, qkv_out: TQKV) -> TQKV:
        pass


class MultiHooks(IHook):
    def __init__(self, hooks: List[IHook]) -> None:
        super().__init__()
        self.hooks = nn.ModuleList(hooks)

    def callback(self, inp: Any, out: Any) -> Any:
        for hook in self.hooks:
            out = hook.callback(inp, out)
        return out
