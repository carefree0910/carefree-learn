import torch

import numpy as np

from typing import *
from cfdata.tabular import DataLoader

from .core import LinearCore
from ..base import ModelBase
from ...types import tensor_dict_type


@ModelBase.register("linear")
class LinearModel(ModelBase):
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
        linear_config = self.config.setdefault("linear_config", {})
        self.core = LinearCore(cfg["in_dim"], cfg["out_dim"], linear_config)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.common_forward(self, batch, batch_indices, loader_name)


__all__ = ["LinearModel"]
