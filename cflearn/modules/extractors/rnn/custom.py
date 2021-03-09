import torch

import torch.nn as nn
import torch.jit as jit

from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from collections import namedtuple

state_type = Tuple[torch.Tensor, torch.Tensor]
return_type = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
LSTMState = namedtuple("LSTMState", ["hx", "cx"])


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, net: torch.Tensor, state: state_type) -> return_type:
        hx, cx = state
        gates = (
            torch.mm(net, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)


class RNNLayer(jit.ScriptModule):
    def __init__(self, cell: Any, input_size: int, hidden_size: int, batch_first: bool):
        super().__init__()
        self.cell = cell(input_size, hidden_size)
        self.batch_first = batch_first

    @jit.script_method
    def forward(self, net: torch.Tensor, state: state_type) -> return_type:
        axis = 1 if self.batch_first else 0
        inputs = net.unbind(axis)
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, dim=axis), state


class LSTM(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool):
        super().__init__()
        self.layer = RNNLayer(LSTMCell, input_size, hidden_size, batch_first)
        self.hidden_size = hidden_size
        self.batch_first = batch_first

    @jit.script_method
    def forward(self, net: torch.Tensor, state: Optional[state_type]) -> return_type:
        if state is None:
            batch = net.shape[0 if self.batch_first else 1]
            zero = torch.zeros(batch, self.hidden_size).to(net)
            state = LSTMState(zero, zero.clone())
        return self.layer(net, state)


__all__ = [
    "LSTM",
]
