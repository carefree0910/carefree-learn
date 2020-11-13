import torch

from typing import Any
from typing import Dict

from ..base import ExtractorBase


rnn_dict = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU, "RNN": torch.nn.RNN}


@ExtractorBase.register("rnn")
class RNN(ExtractorBase):
    def __init__(
        self,
        cell: str,
        in_dim: int,
        cell_config: Dict[str, Any],
        num_layers: int = 1,
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

    @property
    def flatten_ts(self) -> bool:
        return False

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        return net[..., -1, :]


__all__ = ["RNN"]
