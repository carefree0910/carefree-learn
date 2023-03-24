from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from torch.nn import Module


class IHook(Module, metaclass=ABCMeta):
    @abstractmethod
    def callback(self, net: Tensor) -> Tensor:
        pass
