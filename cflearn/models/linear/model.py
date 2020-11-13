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

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.common_forward(self, batch, batch_indices, loader_name)


__all__ = ["LinearModel"]
