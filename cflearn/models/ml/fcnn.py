import torch.nn as nn

from torch import Tensor
from typing import Dict
from typing import List
from typing import Optional
from cftool.misc import safe_execute

from .base import MLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings
from ...modules.blocks import mapping_dict


@MLModel.register("fcnn")
class FCNN(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        input_dim *= num_history
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]
        mapping_base = mapping_dict[mapping_type]
        blocks: List[nn.Module] = []
        for hidden_unit in hidden_units:
            mapping = safe_execute(
                mapping_base,
                dict(
                    in_dim=input_dim,
                    out_dim=hidden_unit,
                    bias=bias,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    rank=rank,
                    rank_ratio=rank_ratio,
                ),
            )
            blocks.append(mapping)
            input_dim = hidden_unit
        blocks.append(nn.Linear(input_dim, output_dim, bias))
        self.hidden_units = hidden_units
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        net = self.encode(net).merged_all
        return self.net(net)


__all__ = ["FCNN"]
