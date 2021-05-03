import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional
from torch.nn.functional import interpolate

from .protocol import DecoderBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import UpsampleConv2d


@DecoderBase.register("vanilla")
class VanillaDecoder(DecoderBase):
    def __init__(
        self,
        img_size: int,
        latent_channels: int,
        latent_resolution: int,
        num_upsample: int,
        out_channels: int,
        last_kernel_size: int = 7,
        norm_type: str = "instance",
        *,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            img_size,
            latent_channels,
            latent_resolution,
            num_upsample,
            out_channels,
            cond_channels=cond_channels,
            num_classes=num_classes,
        )
        self.last_kernel_size = last_kernel_size
        in_nc = latent_channels
        if self.is_conditional:
            in_nc += cond_channels
        blocks: List[nn.Module] = []
        for i in range(self.num_upsample):
            out_nc = (latent_channels if i == 0 else in_nc) // 2
            blocks.extend(
                get_conv_blocks(
                    in_nc,
                    out_nc,
                    3,
                    1,
                    factor=2,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2),
                    conv_base=UpsampleConv2d,
                )
            )
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
        batch = self._inject_cond(batch)
        net = self.decoder(batch[INPUT_KEY])
        net = interpolate(net, size=self.img_size)
        return {PREDICTIONS_KEY: net}


__all__ = ["VanillaDecoder"]
