import torch

from typing import *
from cfdata.tabular import TabularData

from ...bases import ModelBase
from ...misc.toolkit import tensor_dict_type
from ...modules.blocks import MLP


@ModelBase.register("fcnn")
class FCNN(ModelBase):
    def __init__(
        self,
        config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super().__init__(config, tr_data, device)
        self._init_fcnn()

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    def _init_input_config(self):
        super()._init_input_config()
        if self._fc_in_dim > 512:
            hidden_units = [1024, 1024]
        elif self._fc_in_dim > 256:
            if len(self.tr_data) >= 10000:
                hidden_units = [1024, 1024]
            else:
                hidden_units = [2 * self._fc_in_dim, 2 * self._fc_in_dim]
        else:
            num_tr_data = len(self.tr_data)
            if num_tr_data >= 100000:
                hidden_units = [768, 768]
            elif num_tr_data >= 10000:
                hidden_units = [512, 512]
            else:
                hidden_dim = max(64 if num_tr_data >= 1000 else 32, 2 * self._fc_in_dim)
                hidden_units = [hidden_dim, hidden_dim]
        self.hidden_units = self.config.setdefault("hidden_units", hidden_units)
        self.mapping_configs = self.config.setdefault("mapping_configs", {})

    def _init_fcnn(self):
        self._init_input_config()
        final_mapping_config = self.config.setdefault("final_mapping_config", {})
        self.mlp = MLP(
            self._fc_in_dim,
            self._fc_out_dim,
            self.hidden_units,
            self.mapping_configs,
            final_mapping_config=final_mapping_config,
        )

    def forward(self, batch: tensor_dict_type, **kwargs) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch).merge()
        if self.tr_data.is_ts:
            net = net.view(x_batch.shape[0], -1)
        net = self.mlp(net)
        return {"predictions": net}


__all__ = ["FCNN"]
