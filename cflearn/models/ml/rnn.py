import torch.nn as nn
import torch.nn.init as init

from torch import no_grad
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from .fcnn import FCNN
from ..bases import IBAKE
from ..register import register_ml_module
from ..register import register_custom_loss_module
from ...constants import LATENT_KEY
from ...constants import PREDICTIONS_KEY


rnn_dict = {
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "RNN": nn.RNN,
}


@register_ml_module("rnn")
class RNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        cell: str = "GRU",
        num_layers: int = 1,
        hidden_size: int = 256,
        bidirectional: bool = False,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_history = num_history
        rnn_dim = self._init_rnn(cell, num_layers, hidden_size, bidirectional)
        self.head = FCNN(
            rnn_dim,
            output_dim,
            1,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def _init_rnn(
        self,
        cell: str,
        num_layers: int,
        hidden_size: int,
        bidirectional: bool,
    ) -> int:
        rnn_base = rnn_dict[cell]
        input_dimensions = [self.input_dim]
        cell_config = {
            "batch_first": True,
            "hidden_size": hidden_size,
            "bidirectional": bidirectional,
        }
        rnn_dim = hidden_size * (1 + int(bidirectional))
        input_dimensions += [rnn_dim] * (num_layers - 1)
        rnn_list = []
        for dim in input_dimensions:
            rnn = rnn_base(dim, **cell_config)
            with no_grad():
                for name, param in rnn.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
                    elif "bias" in name:
                        init.zeros_(param)
            rnn_list.append(rnn)
        self.rnn_list = nn.ModuleList(rnn_list)
        return rnn_dim

    def forward(self, net: Tensor) -> Tensor:
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        return self.head(net[:, -1])


@register_custom_loss_module("rnn_bake", is_ml=True)
class RNNWithBAKE(IBAKE):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        cell: str = "GRU",
        num_layers: int = 1,
        hidden_size: int = 256,
        bidirectional: bool = False,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        lb: float = 0.1,
        bake_loss: str = "auto",
        bake_loss_config: Optional[Dict[str, Any]] = None,
        w_ensemble: float = 0.5,
        is_classification: bool,
    ):
        super().__init__()
        self.rnn = RNN(
            input_dim,
            output_dim,
            num_history,
            cell,
            num_layers,
            hidden_size,
            bidirectional,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self._init_bake(lb, bake_loss, bake_loss_config, w_ensemble, is_classification)

    def forward(self, net: Tensor) -> tensor_dict_type:  # type: ignore
        for rnn in self.rnn.rnn_list:
            net, final_state = rnn(net, None)
        latent = net[:, -1]
        net = self.rnn.head(latent)
        return {LATENT_KEY: latent, PREDICTIONS_KEY: net}


__all__ = [
    "RNN",
    "RNNWithBAKE",
]
