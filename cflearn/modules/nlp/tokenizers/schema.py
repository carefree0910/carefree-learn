import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from cftool.misc import WithRegister


tokenizers: Dict[str, Type["ITokenizer"]] = {}


class ITokenizer(WithRegister["ITokenizer"], metaclass=ABCMeta):
    d = tokenizers

    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        pass


__all__ = [
    "ITokenizer",
]
