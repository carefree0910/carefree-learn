import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d


class MaskedConv2d(Conv2d):
    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Any = "reflection",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        gain: float = math.sqrt(2.0),
    ):
        assert mask_type in {"A", "B"}
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
            transform_kernel=transform_kernel,
            bias=bias,
            demodulate=demodulate,
            gain=gain,
        )
        self.register_buffer("mask", self.weight.data.clone())
        _, _, h, w = self.weight.shape
        self.mask.fill_(1.0)
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0.0
        self.mask[:, :, h // 2 + 1 :] = 0.0

    def forward(self, x: Tensor, style: Optional[Tensor] = None) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(x, style)


@ModelProtocol.register("pixel_cnn")
class PixelCNN(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_channels: int = 128,
        *,
        num_layers: int = 6,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        in_nc = in_channels
        blocks: List[nn.Module] = []
        for i in range(num_layers):
            mask_type = "A" if i == 0 else "B"
            blocks.extend(
                get_conv_blocks(
                    in_nc,
                    latent_channels,
                    7,
                    1,
                    bias=False,
                    norm_type=norm_type,
                    activation=nn.LeakyReLU(0.2, inplace=True),
                    conv_base=MaskedConv2d,
                    padding=3,
                    mask_type=mask_type,
                )
            )
            in_nc = latent_channels
        blocks.append(Conv2d(latent_channels, num_classes, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.net(batch[INPUT_KEY])}

    def sample(self, num_sample: int, img_size: int) -> Tensor:
        shape = num_sample, self.in_channels, img_size, img_size
        sampled = torch.zeros(shape, device=self.device)
        for i in range(img_size):
            for j in range(img_size):
                out = self.net(sampled)
                probabilities = F.softmax(out[:, :, i, j]).data
                local_sampled = torch.multinomial(probabilities, 1)
                sampled[:, :, i, j] = local_sampled.float() / (self.num_classes - 1)
        return sampled


__all__ = ["PixelCNN"]
