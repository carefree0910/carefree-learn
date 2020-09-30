import torch

from typing import *
from cfdata.tabular import TabularData

from ...bases import ModelBase
from ...misc.toolkit import tensor_dict_type
from ...modules.blocks import Linear


@ModelBase.register("linear")
class LinearModel(ModelBase):
    def __init__(
        self,
        config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super().__init__(config, tr_data, device)
        self._init_linear()

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    def _init_linear(self):
        self._init_input_config()
        self._linear_config = self.config.setdefault("linear_config", {})
        self.linear = Linear(self._fc_in_dim, self._fc_out_dim, **self._linear_config)

    def forward(self, batch: tensor_dict_type, **kwargs) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch).merge()
        if self.tr_data.is_ts:
            net = net.view(x_batch.shape[0], -1)
        net = self.linear(net)
        return {"predictions": net}


__all__ = ["LinearModel"]
