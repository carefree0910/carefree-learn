import math
import torch
import logging

import torch.nn as nn

from typing import *
from cftool.misc import *
from abc import ABCMeta, abstractmethod

encoder_dict: Dict[str, Type["EncoderBase"]] = {}


class EncoderBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(self,
                 idx: int,
                 num_values: int,
                 config: Dict[str, Any]):
        super().__init__()
        self.idx, self.num_values = idx, num_values
        self._init_config(config)

    def _init_config(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def _core(self,
              selected: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self,
                categorical_column: torch.Tensor) -> torch.Tensor:
        selected = categorical_column.to(torch.long)
        # TODO : cache oob_masks for static datasets
        oob_mask = selected >= self.num_values
        if torch.any(oob_mask):
            self.log_msg(
                f"out of bound occurred in categorical column {self.idx}, "
                f"ratio : {torch.mean(oob_mask.to(torch.float)).item():8.6f}",
                self.warning_prefix, 5, logging.WARNING
            )
            selected[oob_mask] = 0
        return self._core(selected)

    @classmethod
    def register(cls, name: str):
        global encoder_dict
        def before(cls_): cls_.__identifier__ = name
        return register_core(name, encoder_dict, before_register=before)
