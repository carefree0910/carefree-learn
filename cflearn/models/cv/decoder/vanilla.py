import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import DecoderBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import UpsampleConv2d


@DecoderBase.register("vanilla")
class VanillaDecoder(DecoderBase):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        norm_type: str = "instance",
        activation: str = "leaky_relu_0.2",
        *,
        num_repeats: Optional[List[int]] = None,
        img_size: Optional[int] = None,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        upsample_conv2d_kwargs: Optional[Dict[str, Any]] = None,
        conv2d_kwargs: Optional[Dict[str, Any]] = None,
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

        if num_repeats is None:
            repeats1 = (self.num_upsample - 1) // 2
            repeats0 = self.num_upsample - repeats1 - 1
            num_repeats = [1] + [4] * repeats0 + [2] * repeats1 + [1]
        if len(num_repeats) != self.num_upsample + 1:
            msg = "length of `num_repeats` is not identical with `num_upsample + 1`"
            raise ValueError(msg)
        in_nc = latent_channels
        if self.is_conditional:
            in_nc += cond_channels

        if conv2d_kwargs is None:
            conv2d_kwargs = {}
        if upsample_conv2d_kwargs is None:
            upsample_conv2d_kwargs = {}

        blocks: List[nn.Module] = []
        for i, num_repeat in enumerate(num_repeats):
            is_last = i == self.num_upsample
            if is_last:
                num_repeat += 1
            if i != 0:
                num_repeat -= 1
                blocks.extend(
                    get_conv_blocks(
                        in_nc,
                        in_nc,
                        3,
                        1,
                        bias=True,
                        factor=2,
                        norm_type=norm_type,
                        activation=activation,
                        conv_base=UpsampleConv2d,
                        padding="reflection",
                        **upsample_conv2d_kwargs,
                    )
                )
            repeat_channels = latent_channels if i == 0 else in_nc
            out_nc = repeat_channels // 2
            for j in range(num_repeat):
                if is_last and j == num_repeat - 1:
                    out_nc = out_channels
                    activation = None
                blocks.extend(
                    get_conv_blocks(
                        in_nc,
                        repeat_channels if j != num_repeat - 1 else out_nc,
                        3,
                        1,
                        bias=True,
                        norm_type=norm_type,
                        activation=activation,
                        padding="reflection",
                        **conv2d_kwargs,
                    )
                )
                in_nc = repeat_channels
            in_nc = out_nc

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
