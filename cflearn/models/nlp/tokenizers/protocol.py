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
from ....misc.toolkit import get_compatible_name


tokenizers: Dict[str, Type["TokenizerProtocol"]] = {}


class TokenizerProtocol(WithRegister["TokenizerProtocol"], metaclass=ABCMeta):
    d = tokenizers

    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "TokenizerProtocol":
        tag = "tokenizers"
        repo = "pretrained-models"
        name = get_compatible_name(tag, repo, name, (3, 8))
        if check_available(tag, repo, name):
            with open(download_tokenizer(name), "rb") as f:
                return dill.load(f)
        return super().make(name, config)


__all__ = [
    "TokenizerProtocol",
]
