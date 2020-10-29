import torch

import torch.nn as nn

from typing import *

from ..fcnn.core import FCNNCore
from ...modules.blocks import DNDF
from ...modules.blocks import TreeResBlock


class TreeDNNCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dndf_input_dim: Optional[int],
        hidden_units: List[int],
        dndf_config: Optional[Dict[str, Any]],
        mapping_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
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


class TreeStackCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_blocks: int,
        dndf_config: Dict[str, Any],
        out_dndf_config: Dict[str, Any],
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(TreeResBlock(in_dim, dndf_config))
        self.out_dndf = DNDF(
            in_dim,
            out_dim,
            **out_dndf_config,
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            net = block(net)
        return self.out_dndf(net)


__all__ = ["TreeDNNCore", "TreeStackCore"]
