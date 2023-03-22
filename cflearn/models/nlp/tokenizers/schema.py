import dill

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from cftool.misc import WithRegister

from ....misc.toolkit import check_available
from ....misc.toolkit import download_tokenizer
from ....misc.toolkit import get_compatible_name


tokenizers: Dict[str, Type["ITokenizer"]] = {}


class ITokenizer(WithRegister["ITokenizer"], metaclass=ABCMeta):
    d = tokenizers

    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        pass

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "ITokenizer":
        if name in tokenizers:
            return super().make(name, config)
        tag = "tokenizers"
        repo = "pretrained-models"
        name = get_compatible_name(tag, repo, name, [(3, 8), (3, 9)])
        if check_available(tag, repo, name):
            with open(download_tokenizer(name), "rb") as f:
                return dill.load(f)
        raise ValueError(f"unrecognized tokenizer '{name}' occurred")


__all__ = [
    "ITokenizer",
]
