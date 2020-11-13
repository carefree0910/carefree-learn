import torch

from ..base import ExtractorBase


@ExtractorBase.register("identity")
class Identity(ExtractorBase):
    @property
    def out_dim(self) -> int:
        return self.transform.out_dim

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return net


__all__ = ["Identity"]
