import torch

import torch.nn as nn

from typing import *

from .rnns import rnn_dict
from ..fcnn.core import FCNNCore


class RNNCore(nn.Module):
    def __init__(
        self,
        cell: str,
        in_dim: int,
        out_dim: int,
        cell_config: Dict[str, Any],
        hidden_units: List[int],
        num_layers: int = 1,
        mapping_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        final_mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # rnn
        rnn_base = rnn_dict[cell]
        input_dimensions = [in_dim]
        hidden_size = cell_config["hidden_size"]
        input_dimensions += [hidden_size] * (num_layers - 1)
        self.rnn_list = torch.nn.ModuleList(
            [rnn_base(dim, **cell_config) for dim in input_dimensions]
        )
        # fcnn
        self.fcnn = FCNNCore(
            hidden_size,
            out_dim,
            hidden_units,
            mapping_configs,
            final_mapping_config,
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        return self.fcnn(net[..., -1, :])


__all__ = ["RNNCore"]
