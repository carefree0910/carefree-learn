import torch.nn as nn

from torch import Tensor

from .protocols import FFN
from ..convs import DepthWiseConv2d
from ..common import Lambda
from ..customs import Linear
from ..activations import Activation


@FFN.register("ff")
class FeedForward(FFN):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        dropout: float,
        activation: str = "GELU",
    ):
        super().__init__(in_dim, latent_dim, dropout)
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            Activation.make(activation),
            nn.Dropout(dropout),
            Linear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    @property
    def need_2d(self) -> bool:
        return False

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


@FFN.register("mix_ff")
class MixFeedForward(FFN):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__(in_dim, latent_dim, dropout)
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            Lambda(lambda t: t.permute(0, 3, 1, 2), "permute -> BCHW"),
            DepthWiseConv2d(latent_dim),
            Lambda(lambda t: t.flatten(2).transpose(1, 2), "transpose -> BNC"),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    @property
    def need_2d(self) -> bool:
        return True

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


__all__ = [
    "FeedForward",
    "MixFeedForward",
]
