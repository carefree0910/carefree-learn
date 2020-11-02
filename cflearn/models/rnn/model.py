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
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__(
            pipeline_config,
            tr_loader,
            cv_loader,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        cfg = self.get_core_config(self)
        self.core = RNNCore(**cfg)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

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
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch, batch_indices, loader_name).merge()
        return {"predictions": self.core(net)}


__all__ = ["RNN"]
