from abc import ABCMeta
from typing import Any

from ...protocol import ModelProtocol


class MLModelProtocol(ModelProtocol, metaclass=ABCMeta):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim


__all__ = [
    "MLModelProtocol",
]
