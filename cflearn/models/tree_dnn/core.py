import torch

import torch.nn as nn

from typing import *

from ..fcnn.core import FCNNCore
from ...modules.blocks import DNDF


class TreeDNNCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dndf_input_dim: Optional[int],
        hidden_units: List[int],
        dndf_config: Optional[Dict[str, Any]],
        mapping_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        final_mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.fcnn = FCNNCore(
            in_dim,
            out_dim,
            hidden_units,
            mapping_configs,
            final_mapping_config,
        )
        if dndf_config is None:
            self.dndf = None
        else:
            if dndf_input_dim is None:
                raise ValueError("`dndf_input_dim` should be used when `dndf` is used")
            self.dndf = DNDF(dndf_input_dim, out_dim, **dndf_config)

    def forward(
        self,
        fcnn_net: torch.Tensor,
        dndf_net: Optional[torch.Tensor],
    ) -> torch.Tensor:
        fcnn_net = self.fcnn(fcnn_net)
        if self.dndf is None:
            return fcnn_net
        assert dndf_net is not None
        dndf_net = self.dndf(dndf_net)
        return fcnn_net + dndf_net


__all__ = ["TreeDNNCore"]
