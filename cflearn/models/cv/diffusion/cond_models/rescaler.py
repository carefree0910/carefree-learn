import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from functools import partial

from .schema import ISpecializedConditionModel
from .....modules.blocks import HijackConv2d


@ISpecializedConditionModel.register("rescaler")
class SpatialRescaler(ISpecializedConditionModel):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_stages: int = 1,
        multiplier: float = 0.5,
        method: str = "bilinear",
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.factor = round(1.0 / multiplier**num_stages)
        supported = {"nearest", "linear", "bilinear", "trilinear", "bicubic", "area"}
        if method not in supported:
            raise ValueError(f"`method` should be one of {supported}")
        self.interpolator = partial(F.interpolate, mode=method, scale_factor=multiplier)
        if out_channels is None:
            self.channel_mapper = None
        else:
            self.channel_mapper = HijackConv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, cond: Tensor) -> Tensor:
        for _ in range(self.num_stages):
            cond = self.interpolator(cond)
        if self.channel_mapper is not None:
            cond = self.channel_mapper(cond)
        return cond


__all__ = [
    "SpatialRescaler",
]
