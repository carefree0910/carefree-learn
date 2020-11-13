import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import Optional
from cfdata.tabular import DataLoader

from .core import TransformerCore
from ..base import ModelBase
from ..fcnn import FCNN
from ...types import tensor_dict_type


@ModelBase.register("transformer")
class Transformer(ModelBase):
    def define_heads(self) -> None:
        cfg = self.get_core_config(self)
        self.add_head("basic", TransformerCore(**cfg))

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        in_dim = instance.tr_data.processed_dim
        transformer_config = instance.config.setdefault("transformer_config", {})
        latent_dim = transformer_config.setdefault("latent_dim", None)
        transformer_config.setdefault("to_latent", latent_dim is not None)
        il_config = transformer_config.setdefault("input_linear_config", None)
        if il_config is not None:
            il_config.setdefault("bias", False)
        transformer_config.setdefault("num_layers", 6)
        transformer_config.setdefault("num_heads", 8)
        transformer_config.setdefault("norm", None)
        transformer_config.setdefault("transformer_layer_config", {})
        cfg = FCNN.get_core_config(instance)
        cfg.update(transformer_config)
        cfg["num_history"] = instance.num_history
        cfg["in_dim"] = in_dim
        return cfg

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        self.common_forward(self, batch, batch_indices, loader_name)


__all__ = ["Transformer"]
