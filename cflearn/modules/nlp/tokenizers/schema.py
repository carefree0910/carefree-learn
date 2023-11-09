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

from ....toolkit import check_available
from ....toolkit import download_tokenizer
from ....toolkit import get_compatible_name
from ....toolkit import DownloadDtype


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
        name = get_compatible_name(DownloadDtype.TOKENIZERS, name, [(3, 8), (3, 9)])
        if check_available(DownloadDtype.TOKENIZERS, name):
            with open(download_tokenizer(name), "rb") as f:
                return dill.load(f)
        raise ValueError(f"unrecognized tokenizer '{name}' occurred")


__all__ = [
    "ITokenizer",
]
