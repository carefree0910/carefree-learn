import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..protocol import ImageTranslatorMixin
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ..encoder.backbone import BackboneEncoder
from ....misc.toolkit import interpolate
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Linear


@ModelProtocol.register("linear_seg")
class LinearSegmentation(ModelProtocol, ImageTranslatorMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 768,
        *,
        dropout: float = 0.1,
        backbone: str = "mix_vit",
        backbone_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = BackboneEncoder(
            backbone,
            in_channels,
            backbone_config=backbone_config,
        )
        linear_blocks: List[nn.Module] = []
        increment_config = self.backbone.net.increment_config
        backbone_channels = increment_config["out_channels"]
        for num_channel in backbone_channels:
            linear_blocks.append(
                nn.Sequential(
                    Lambda(lambda t: t.flatten(2).transpose(1, 2), "BCHW -> BNC"),
                    Linear(num_channel, latent_dim),
                )
            )
        self.linear_blocks = nn.ModuleList(linear_blocks)
        self.linear_fuse = nn.Sequential(
            *get_conv_blocks(
                latent_dim * 4,
                latent_dim,
                kernel_size=1,
                stride=1,
                norm_type="batch",
            )
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.linear_head = Conv2d(latent_dim, out_channels, kernel_size=1)

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
        outputs = []
        for i, linear in enumerate(self.linear_blocks):
            net = features[f"stage{i + 1}"]
            b, _, h, w = net.shape
            net = linear(net).transpose(1, 2)
            net = net.contiguous().view(b, -1, h, w)
            net = interpolate(net, anchor=inp, mode="bilinear")
            outputs.append(net)
        net = torch.cat(outputs, dim=1)
        net = self.linear_fuse(net)
        if self.dropout is not None:
            net = self.dropout(net)
        net = self.linear_head(net)
        return {PREDICTIONS_KEY: net}


__all__ = ["LinearSegmentation"]
