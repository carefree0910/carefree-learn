import torch

from typing import Any
from typing import Dict
from torch.nn import init

from .custom import *
from ..base import ExtractorBase
from ...transform.core import Dimensions


rnn_dict = {
    "LSTM": torch.nn.LSTM,
    "GRU": torch.nn.GRU,
    "RNN": torch.nn.RNN,
    "JitLSTM": LSTM,
}


@ExtractorBase.register("rnn")
class RNN(ExtractorBase):
    def __init__(
        self,
        in_flat_dim: int,
        dimensions: Dimensions,
        cell: str,
        cell_config: Dict[str, Any],
        num_layers: int = 1,
    ):
        super().__init__(in_flat_dim, dimensions)
        # rnn
        rnn_base = rnn_dict[cell]
        input_dimensions = [self.in_dim]
        self.hidden_size = cell_config["hidden_size"]
        self.bidirectional = cell_config.setdefault("bidirectional", False)
        input_dimensions += [self.hidden_size] * (num_layers - 1)
        rnn_list = []
        for dim in input_dimensions:
            rnn = rnn_base(dim, **cell_config)
            with torch.no_grad():
                for name, param in rnn.named_parameters():
                    if "weight" in name:
                        init.orthogonal_(param)
                    elif "bias" in name:
                        init.zeros_(param)
            rnn_list.append(rnn)
        self.rnn_list = torch.nn.ModuleList(rnn_list)

    @property
    def flatten_ts(self) -> bool:
        return False

    @property
    def out_dim(self) -> int:
        if not self.bidirectional:
            return self.hidden_size
        return 2 * self.hidden_size

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for rnn in self.rnn_list:
            net, final_state = rnn(net, None)
        return net[..., -1, :]


__all__ = ["RNN"]
