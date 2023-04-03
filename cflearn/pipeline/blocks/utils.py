import os

from abc import abstractmethod
from abc import ABCMeta
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict

from ..core import Block
from ...schema import DLConfig


class InjectDefaultsMixin:
    _defaults: OrderedDictType

    def __init__(self) -> None:
        self._defaults = OrderedDict()

    def process_defaults(self, _defaults: OrderedDictType) -> None:
        for k, v in self._defaults.items():
            _defaults[k] = v


class TryLoadBlock(Block, metaclass=ABCMeta):
    # abstract

    @abstractmethod
    def try_load(self, folder: str) -> bool:
        pass

    @abstractmethod
    def from_scratch(self, config: DLConfig) -> None:
        pass

    @abstractmethod
    def dump_to(self, folder: str) -> None:
        pass

    # inheritance

    def build(self, config: DLConfig) -> None:
        if self.serialize_folder is not None:
            serialize_folder = os.path.join(self.serialize_folder, self.__identifier__)
            if self.try_load(serialize_folder):
                return
        self.from_scratch(config)

    def save_extra(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        self.dump_to(folder)


__all__ = [
    "InjectDefaultsMixin",
    "TryLoadBlock",
]
