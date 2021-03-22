from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from cftool.misc import register_core

from ...types import tensor_dict_type


aggregator_dict: Dict[str, Type["AggregatorBase"]] = {}


class AggregatorBase(metaclass=ABCMeta):
    def __init__(self, **kwargs: Any):
        self.config = kwargs

    @abstractmethod
    def reduce(self, outputs: tensor_dict_type, **kwargs: Any) -> Tensor:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global aggregator_dict
        return register_core(name, aggregator_dict)

    @classmethod
    def make(cls, name: str, **kwargs: Any) -> "AggregatorBase":
        return aggregator_dict[name](**kwargs)


__all__ = ["AggregatorBase"]
