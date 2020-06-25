import torch
import torch.nn as nn

from typing import *
from cfdata.tabular import TabularData

from ...bases import ModelBase
from ...modules.blocks import *


@ModelBase.register("fcnn")
class FCNN(ModelBase):
    def __init__(self,
                 config: Dict[str, Any],
                 tr_data: TabularData,
                 device: torch.device):
        super().__init__(config, tr_data, device)
        self._init_fcnn()

    def _init_config(self,
                     config: Dict[str, Any],
                     tr_data: TabularData):
        super()._init_config(config, tr_data)
        self._fc_in_dim, self._fc_out_dim = map(self.config.get, ["fc_in_dim", "fc_out_dim"])
        self.hidden_units = self.config.setdefault("hidden_units", [256, 256])
        self.mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(self.mapping_configs, dict):
            self.mapping_configs = [self.mapping_configs] * len(self.hidden_units)
        self.out_dim = max(tr_data.num_classes, 1)

    def _init_fcnn(self):
        mappings = []
        if self._fc_in_dim is not None:
            in_dim = self._fc_in_dim
        else:
            in_dim = self._fc_in_dim = self.merged_dim
        if self._fc_out_dim is not None:
            out_dim = self._fc_out_dim
        else:
            out_dim = self._fc_out_dim = self.out_dim
        for i, hidden_units in enumerate(self.hidden_units):
            mapping_config = self.mapping_configs[i]
            mappings.append(Mapping(in_dim, hidden_units, **mapping_config))
            in_dim = hidden_units
        final_layer_config = self.config.setdefault("final_layer_config", {})
        mappings.append(Linear(in_dim, out_dim, **final_layer_config))
        tuple(map(Mapping.reset_parameters, mappings))
        self.mappings = nn.ModuleList(mappings)

    def forward(self,
                batch: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch).merge()
        for mapping in self.mappings:
            net = mapping(net)
        return {"predictions": net}


__all__ = ["FCNN"]
