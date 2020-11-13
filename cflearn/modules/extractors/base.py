import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from cftool.misc import register_core

from ..transform.core import Transform


extractor_dict: Dict[str, Type["ExtractorBase"]] = {}


class ExtractorBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, transform: Transform, **kwargs: Any):
        super().__init__()
        self.transform = transform

    @property
    def flatten_ts(self) -> bool:
        return True

    @property
    @abstractmethod
    def out_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global extractor_dict
        return register_core(name, extractor_dict)

    @classmethod
    def make(
        cls,
        name: str,
        transform: Transform,
        config: Dict[str, Any],
    ) -> "ExtractorBase":
        return extractor_dict[name](transform, **config)


__all__ = ["ExtractorBase"]
