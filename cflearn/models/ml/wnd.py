import torch

from typing import Any
from typing import List
from typing import Optional
from cftool.misc import shallow_copy_dict

from .fcnn import FCNN
from .linear import Linear
from .protocol import ONE_HOT_KEY
from .protocol import EMBEDDING_KEY
from .protocol import NUMERICAL_KEY
from .protocol import MLCoreProtocol
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY


@MLCoreProtocol.register("wnd")
class WideAndDeep(MLCoreProtocol):
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
        super().__init__(in_dim, out_dim, num_history)
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
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        one_hot = batch[ONE_HOT_KEY]
        embedding = batch[EMBEDDING_KEY]
        numerical = batch[NUMERICAL_KEY]
        # wide
        wide_batch = shallow_copy_dict(batch)
        if one_hot is None or embedding is None:
            wide_batch[INPUT_KEY] = numerical
        else:
            wide_batch[INPUT_KEY] = torch.cat([one_hot, embedding], dim=-1)
        wide_net = self.linear(batch_idx, wide_batch, state, **kwargs)[PREDICTIONS_KEY]
        # deep
        deep_batch = shallow_copy_dict(batch)
        if embedding is None:
            deep_batch[INPUT_KEY] = numerical
        else:
            deep_batch[INPUT_KEY] = torch.cat([numerical, embedding], dim=-1)
        deep_net = self.fcnn(batch_idx, deep_batch, state, **kwargs)[PREDICTIONS_KEY]
        # merge
        return {PREDICTIONS_KEY: wide_net + deep_net}


__all__ = ["WideAndDeep"]
