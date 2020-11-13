import torch

import numpy as np

from typing import *
from cfdata.tabular import DataLoader

from .core import RNNCore
from ..base import ModelBase
from ..fcnn import FCNN
from ...types import tensor_dict_type


@ModelBase.register("rnn")
class RNN(ModelBase):
    def define_heads(self) -> None:
        cfg = self.get_core_config(self)
        self.add_head("basic", RNNCore(**cfg))

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        rnn_config = instance.config.setdefault("rnn_config", {})
        rnn_config.setdefault("cell", "GRU")
        rnn_config.setdefault("num_layers", 1)
        cell_config = rnn_config.setdefault("cell_config", {})
        cell_config["batch_first"] = True
        cell_config.setdefault("hidden_size", 256)
        cell_config.setdefault("bidirectional", False)
        cfg = FCNN.get_core_config(instance)
        cfg["in_dim"] = instance.tr_data.processed_dim
        cfg.update(rnn_config)
        return cfg

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.common_forward(self, batch, batch_indices, loader_name)


__all__ = ["RNN"]
