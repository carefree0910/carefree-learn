import dill

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from ....misc.toolkit import WithRegister
from ....misc.toolkit import check_available
from ....misc.toolkit import download_tokenizer


tokenizers: Dict[str, Type["TokenizerProtocol"]] = {}


class TokenizerProtocol(WithRegister, metaclass=ABCMeta):
    d = tokenizers

    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "TokenizerProtocol":
        if check_available("tokenizers", "pretrained-models", name):
            with open(download_tokenizer(name), "rb") as f:
                return dill.load(f)
        return super().make(name, config)


__all__ = [
    "TokenizerProtocol",
]
