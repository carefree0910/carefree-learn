import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from .protocol import DecoderBase
from ..toolkit import auto_num_downsample
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import UpsampleConv2d


@DecoderBase.register("vanilla")
class VanillaDecoder(DecoderBase):
    def __init__(
        self,
        img_size: int,
        latent_channels: int,
        out_channels: int,
        last_kernel_size: int = 7,
    ):
        super().__init__(img_size, latent_channels, out_channels)
        self.last_kernel_size = last_kernel_size
        self.num_upsample = auto_num_downsample(img_size)
        blocks: List[nn.Module] = []
        in_nc = latent_channels
        for i in range(self.num_upsample):
            out_nc = in_nc // 2
            blocks.append(UpsampleConv2d(in_nc, out_nc, factor=2, kernel_size=3))
            in_nc = out_nc
        blocks.append(
            Conv2d(
                in_nc,
                out_channels,
                kernel_size=last_kernel_size,
                stride=1,
            )
        )
        self.decoder = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.decoder(batch[INPUT_KEY])}


__all__ = ["VanillaDecoder"]
