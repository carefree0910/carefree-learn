import torch.nn as nn
import torch.nn.init as init

from torch import no_grad
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .fcnn import FCNN
from .protocol import MERGED_KEY
from .protocol import MLCoreProtocol
from ..bases import BAKEBase
from ...types import tensor_dict_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...constants import LATENT_KEY


rnn_dict = {
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "RNN": nn.RNN,
}


@MLCoreProtocol.register("rnn")
class RNN(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
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
        super().__init__(in_dim, out_dim, num_history)
        rnn_dim = self._init_rnn(cell, num_layers, hidden_size, bidirectional)
        self.head = FCNN(
            rnn_dim,
            out_dim,
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
        input_dimensions = [self.in_dim]
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

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[MERGED_KEY]
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        batch[MERGED_KEY] = net[:, -1]
        return self.head(batch_idx, batch, state, **kwargs)


@MLCoreProtocol.register("bake_rnn")
class RNNWithBAKE(BAKEBase, RNN):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
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
        RNN.__init__(
            self,
            in_dim,
            out_dim,
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
        # BAKE
        self.lb = lb
        self.w_ensemble = w_ensemble
        self.is_classification = is_classification
        if bake_loss == "auto":
            bake_loss = "focal" if is_classification else "mae"
        self.bake_loss = LossProtocol.make(bake_loss, config=bake_loss_config or {})

    def forward_with_latent(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        latent = net[:, -1]
        batch[MERGED_KEY] = latent
        results = self.head(batch_idx, batch, state, **kwargs)
        results[LATENT_KEY] = latent
        return results


__all__ = ["RNN"]
