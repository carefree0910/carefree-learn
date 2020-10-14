import math
import torch
import logging

import torch.nn as nn

from typing import *
from abc import ABCMeta, abstractmethod
from cftool.misc import LoggingMixin, register_core

from ..misc.toolkit import tensor_dict_type, Initializer

encoder_dict: Dict[str, Type["EncoderBase"]] = {}


class EncoderBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        idx: int,
        num_values: int,
        config: Dict[str, Any],
    ):
        super().__init__()
        self.idx, self.num_values = idx, num_values
        self._init_config(config)

    def _init_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def _core(self, selected: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, categorical_column: torch.Tensor) -> torch.Tensor:
        selected = categorical_column.to(torch.long)
        # TODO : cache oob_masks for static datasets
        oob_mask = selected >= self.num_values
        if torch.any(oob_mask):
            self.log_msg(
                f"out of bound occurred in categorical column {self.idx}, "
                f"ratio : {torch.mean(oob_mask.to(torch.float)).item():8.6f}",
                prefix=self.warning_prefix,
                verbose_level=5,
                msg_level=logging.WARNING,
            )
            # TODO : currently pytorch does not support onnx with bool masks
            #        in the future this line should be un-indented
            selected[oob_mask] = 0
        return self._core(selected)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global encoder_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, encoder_dict, before_register=before)


@EncoderBase.register("one_hot")
class OneHot(EncoderBase):
    @property
    def dim(self) -> int:
        return self.num_values

    def _core(self, selected: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(selected, num_classes=self.num_values)
        return one_hot.to(torch.float32)


@EncoderBase.register("embedding")
class Embedding(EncoderBase):
    def __init__(
        self,
        idx: int,
        num_values: int,
        config: Dict[str, Any],
    ):
        super().__init__(idx, num_values, config)
        self.embedding = nn.Embedding(num_values, self._dim)
        if self._init_method is not None:
            embedding_initializer = Initializer(self._init_config_)
            embedding_initializer.initialize(self.embedding.weight, self._init_method)

    def _init_config(self, config: Dict[str, Any]) -> None:
        super()._init_config(config)
        self._init_method = self.config.setdefault("init_method", "truncated_normal")
        self._init_config_ = self.config.setdefault(
            "init_config", {"mean": 0.0, "std": 0.02}
        )
        embedding_dim = self.config.setdefault("embedding_dim", "auto")
        if isinstance(embedding_dim, int):
            self._dim = embedding_dim
        elif embedding_dim == "log":
            self._dim = math.ceil(math.log2(self.num_values))
        elif embedding_dim == "sqrt":
            self._dim = math.ceil(math.sqrt(self.num_values))
        elif embedding_dim == "auto":
            self._dim = min(
                self.num_values, max(4, min(8, math.ceil(math.log2(self.num_values))))
            )
        else:
            raise ValueError(f"embedding dim '{embedding_dim}' is not defined")

    @property
    def dim(self) -> int:
        return self._dim

    def _core(self, selected: torch.Tensor) -> torch.Tensor:
        return self.embedding(selected)


class EncoderStack(nn.Module, LoggingMixin):
    def __init__(self, *encoders: EncoderBase):
        super().__init__()
        encoders_: Dict[str, EncoderBase] = {}
        for encoder in encoders:
            key = encoder.__identifier__
            if key in encoders_:
                raise ValueError(f"'{key}' encoder is already stacked")
            encoders_[key] = encoder
        self.encoders = nn.ModuleDict(encoders_)
        self.sorted_keys = sorted(encoders_.keys())

    @property
    def dim(self) -> int:
        return sum(self.dims.values())

    @property
    def dims(self) -> Dict[str, int]:
        return {k: encoder.dim for k, encoder in self.encoders.items()}

    def forward(
        self,
        categorical_column: torch.Tensor,
        *,
        return_all: bool = False,
    ) -> Union[torch.Tensor, tensor_dict_type]:
        encodings = {k: v(categorical_column) for k, v in self.encoders.items()}
        if return_all:
            return encodings
        return torch.cat([encodings[k] for k in self.sorted_keys], dim=1)


__all__ = ["EncoderBase", "EncoderStack", "encoder_dict"]
