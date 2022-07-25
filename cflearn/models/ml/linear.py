import torch

from ..register import register_ml_module


@register_ml_module("linear")
class Linear(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        *,
        bias: bool = True,
    ):
        super().__init__()
        self.net = torch.nn.Linear(input_dim * num_history, output_dim, bias)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.net(net)


__all__ = ["Linear"]
