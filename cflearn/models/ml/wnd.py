import torch

from typing import Any
from typing import List
from typing import Optional

from .fcnn import FCNN
from .linear import Linear
from .protocol import ONE_HOT_KEY
from .protocol import EMBEDDING_KEY
from .protocol import NUMERICAL_KEY
from .protocol import register_ml_module
from .protocol import Dimensions
from ...types import tensor_dict_type
from ...protocol import TrainerState


@register_ml_module("wnd", use_full_input=True)
class WideAndDeep(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fcnn = FCNN(
            in_dim,
            out_dim,
            num_history,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self.linear = Linear(in_dim, out_dim, num_history, bias=bias)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        one_hot = batch[ONE_HOT_KEY]
        embedding = batch[EMBEDDING_KEY]
        numerical = batch[NUMERICAL_KEY]
        # wide
        if one_hot is None or embedding is None:
            wide_inp = numerical
        else:
            wide_inp = torch.cat([one_hot, embedding], dim=-1)
        wide_net = self.linear(wide_inp)
        # deep
        if embedding is None:
            deep_inp = numerical
        else:
            deep_inp = torch.cat([numerical, embedding], dim=-1)
        deep_net = self.fcnn(deep_inp)
        # merge
        return wide_net + deep_net


__all__ = ["WideAndDeep"]
