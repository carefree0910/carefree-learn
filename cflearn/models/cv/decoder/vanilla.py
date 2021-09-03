import torch.nn as nn

from typing import Any
from typing import List
from typing import Optional

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
        latent_channels: int,
        out_channels: int,
        last_kernel_size: int = 7,
        norm_type: str = "instance",
        *,
        img_size: Optional[int] = None,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
    ):
        super().__init__(
            latent_channels,
            out_channels,
            img_size=img_size,
            num_upsample=num_upsample,
            cond_channels=cond_channels,
            num_classes=num_classes,
            latent_resolution=latent_resolution,
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
                    bias=False,
                    factor=2,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2, inplace=True),
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
                bias=False,
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
        net = self.resize(net)
        return {PREDICTIONS_KEY: net}


__all__ = ["VanillaDecoder"]
