import torch

import torch.nn as nn

from typing import *

from ...modules.blocks import MLP


class FCNNCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_units: List[int],
        mapping_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        final_mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.mlp = MLP(
            in_dim,
            out_dim,
            hidden_units,
            mapping_configs,
            final_mapping_config=final_mapping_config,
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.mlp(net)


__all__ = ["FCNNCore"]
