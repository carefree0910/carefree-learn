import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from cftool.misc import register_core


head_dict: Dict[str, Type["HeadBase"]] = {}


class HeadBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        pass

    def _init_config(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global head_dict
        return register_core(name, head_dict)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "HeadBase":
        return head_dict[name](**config)


__all__ = ["HeadBase"]
