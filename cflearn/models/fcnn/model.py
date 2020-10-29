import torch

import numpy as np

from typing import *
from cfdata.tabular import DataLoader

from .core import FCNNCore
from ..base import ModelBase
from ...types import tensor_dict_type


@ModelBase.register("fcnn")
class FCNN(ModelBase):
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
        cfg = self.get_input_config(self)
        self.core = FCNNCore(**cfg)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    @staticmethod
    def get_input_config(instance: "ModelBase") -> Dict[str, Any]:
        cfg = ModelBase.get_input_config(instance)
        in_dim: int = cfg["in_dim"]
        if in_dim > 512:
            hidden_units = [1024, 1024]
        elif in_dim > 256:
            if len(instance.tr_data) >= 10000:
                hidden_units = [1024, 1024]
            else:
                hidden_units = [2 * in_dim, 2 * in_dim]
        else:
            num_tr_data = len(instance.tr_data)
            if num_tr_data >= 100000:
                hidden_units = [768, 768]
            elif num_tr_data >= 10000:
                hidden_units = [512, 512]
            else:
                hidden_dim = max(64 if num_tr_data >= 1000 else 32, 2 * in_dim)
                hidden_units = [hidden_dim, hidden_dim]
        hidden_units = instance.config.setdefault("hidden_units", hidden_units)
        mapping_configs = instance.config.setdefault("mapping_configs", {})
        fm_config = instance.config.setdefault("final_mapping_config", {})
        cfg.update(
            {
                "hidden_units": hidden_units,
                "mapping_configs": mapping_configs,
                "final_mapping_config": fm_config,
            }
        )
        return cfg

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.common_forward(self, batch, batch_indices, loader_name)


__all__ = ["FCNN"]
