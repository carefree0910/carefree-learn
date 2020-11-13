import numpy as np

from typing import *

from .core import LinearCore
from ..base import ModelBase
from ...types import tensor_dict_type


@ModelBase.register("linear")
class LinearModel(ModelBase):
    def define_heads(self) -> None:
        cfg = self.get_core_config(self)
        linear_config = self.config.setdefault("linear_config", {})
        core = LinearCore(cfg["in_dim"], cfg["out_dim"], linear_config)
        self.add_head("basic", core)


__all__ = ["LinearModel"]
