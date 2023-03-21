import torch.nn as nn
import torch.nn.init as init

from torch import no_grad
from torch import Tensor
from typing import Dict
from typing import List
from typing import Optional

from .fcnn import FCNN
from typing import Dict
from typing import Optional

from .base import MLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings


rnn_dict = {
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "RNN": nn.RNN,
}


@MLModel.register("rnn")
class RNN(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
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
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
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


__all__ = [
    "RNN",
]
