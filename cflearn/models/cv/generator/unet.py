import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ..encoder.backbone import BackboneEncoder1D
from ....misc.toolkit import interpolate
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d


class UNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.net = nn.Sequential(
            *get_conv_blocks(
                in_channels + skip_channels,
                out_channels,
                3,
                1,
                norm_type=norm_type,
                activation=nn.LeakyReLU(0.2, inplace=True),
                padding=1,
            ),
            *get_conv_blocks(
                out_channels,
                out_channels,
                3,
                1,
                norm_type=norm_type,
                activation=nn.LeakyReLU(0.2, inplace=True),
                padding=1,
            ),
        )

    def forward(self, net: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        net = interpolate(net, mode="nearest", factor=2)
        if skip is not None:
            net = torch.cat([skip, interpolate(net, anchor=skip)], dim=1)
        return self.net(net)


class UNetDecoder(nn.Module):
    def __init__(self, backbone_channels: List[int], **kwargs: Any):
        super().__init__()
        in_channels = backbone_channels[::-1]
        skip_channels = in_channels[1:] + [0]
        out_channels = skip_channels.copy()
        out_channels[-1] = in_channels[-1]
        blocks = [
            UNetDecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, backbone_outputs: tensor_dict_type) -> Tensor:
        features = [backbone_outputs[f"stage{i}"] for i in range(len(backbone_outputs))]
        features = features[::-1]

        net = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            net = decoder_block(net, skip)

        return net


@ModelProtocol.register("unet")
class UNet(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str = "mobilenet_v2",
        backbone_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        # `backbone` here should define `out_channels` in its `increment_config`
        super().__init__()
        if backbone_config is None:
            backbone_config = {}
        self.backbone = BackboneEncoder1D(backbone, in_channels, **backbone_config)
        increment_config = self.backbone.net.increment_config
        if increment_config is None:
            raise ValueError(
                "`increment_config` should be provided in `BackboneEncoder` "
                "for `UNetGenerator`"
            )
        backbone_channels = increment_config.get("out_channels")
        if backbone_channels is None:
            raise ValueError(
                "`out_channels` should be provided in `increment_config` "
                "for `UNetGenerator`"
            )
        self.decoder = UNetDecoder(backbone_channels, **(decoder_config or {}))
        self.head = Conv2d(backbone_channels[0], out_channels, kernel_size=3)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        inp = batch[INPUT_KEY]
        features = self.backbone(batch_idx, batch, state, **kwargs)
        features.pop(LATENT_KEY)
        net = self.head(self.decoder(features))
        return {PREDICTIONS_KEY: interpolate(net, anchor=inp)}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]


__all__ = ["UNet"]
