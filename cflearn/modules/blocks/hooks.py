from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Tuple
from torch.nn import Module


TQKV = Tuple[Tensor, Tensor, Tensor]


class IHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, inp: Tensor, out: Tensor) -> Tensor:
        pass


class IAttentionHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, qkv_inp: TQKV, qkv_out: TQKV) -> TQKV:
        pass
