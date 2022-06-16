import torch

from .protocol import register_ml_module


@register_ml_module("linear")
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        *,
        bias: bool = True,
    ):
        super().__init__()
        in_dim *= num_history
        self.net = torch.nn.Linear(in_dim * num_history, out_dim, bias)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.net(net)


__all__ = ["Linear"]
