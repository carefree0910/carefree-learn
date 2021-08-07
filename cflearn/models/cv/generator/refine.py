import torch.nn as nn

from typing import Any
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import ResidualBlock


@ModelProtocol.register("alpha_refine")
class AlphaRefineNet(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int = 3,
        latent_channels: int = 64,
        dropout: float = 0.0,
        eca_kernel_size: Optional[int] = None,
    ):
        super().__init__()
        blocks = get_conv_blocks(
            in_channels,
            latent_channels,
            3,
            1,
            activation=nn.ReLU(inplace=True),
        )
        for _ in range(num_layers - 2):
            blocks.append(
                ResidualBlock(
                    latent_channels,
                    dropout,
                    eca_kernel_size=eca_kernel_size,
                )
            )
        blocks.extend(get_conv_blocks(latent_channels, out_channels, 3, 1))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.net(batch[INPUT_KEY])}


__all__ = [
    "AlphaRefineNet",
]
