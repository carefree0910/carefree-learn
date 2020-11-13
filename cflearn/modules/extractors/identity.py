import torch

from .base import ExtractorBase


@ExtractorBase.register("identity")
class Identity(ExtractorBase):
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return net


__all__ = ["Identity"]
