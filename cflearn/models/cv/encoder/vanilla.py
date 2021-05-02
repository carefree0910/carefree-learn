import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

from .protocol import EncoderBase
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d


@EncoderBase.register("vanilla")
class VanillaEncoder(EncoderBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_downsample: int,
        latent_channels: int = 128,
        first_kernel_size: int = 7,
    ):
        super().__init__(img_size, in_channels, num_downsample, latent_channels)
        self.first_kernel_size = first_kernel_size
        start_channels = int(round(latent_channels / (2 ** self.num_downsample)))
        if start_channels <= 0:
            raise ValueError(
                f"latent_channels ({latent_channels}) is too small "
                f"for num_downsample ({self.num_downsample})"
            )
        blocks = get_conv_blocks(
            in_channels,
            start_channels,
            first_kernel_size,
            1,
            activation=nn.LeakyReLU(0.2),
        )
        in_nc = start_channels
        for i in range(self.num_downsample):
            is_last = i == self.num_downsample - 1
            if is_last:
                out_nc = latent_channels
            else:
                out_nc = min(in_nc * 2, latent_channels)
            new_blocks: List[nn.Module]
            if is_last:
                new_blocks = [Conv2d(in_nc, out_nc, kernel_size=3, stride=2)]
            else:
                new_blocks = get_conv_blocks(
                    in_nc,
                    out_nc,
                    3,
                    2,
                    activation=nn.LeakyReLU(0.2),
                )
            blocks.extend(new_blocks)
            in_nc = out_nc
        self.encoder = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.encoder(batch[INPUT_KEY])}


__all__ = ["VanillaEncoder"]
