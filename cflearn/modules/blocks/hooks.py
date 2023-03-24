from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from torch.nn import Module


class LinearHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, net: Tensor) -> Tensor:
        pass


class ConvHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, net: Tensor) -> Tensor:
        pass
