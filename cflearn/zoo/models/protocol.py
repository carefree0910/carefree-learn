from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from cftool.misc import WithRegister


dl_zoo_model_loaders: Dict[str, Type["IDLZooModelLoader"]] = {}


class IDLZooModelLoader(WithRegister, metaclass=ABCMeta):
    d = dl_zoo_model_loaders

    @abstractmethod
    def permute_kwargs(self, kwargs: Dict[str, Any]) -> None:
        pass


__all__ = [
    "dl_zoo_model_loaders",
    "IDLZooModelLoader",
]
