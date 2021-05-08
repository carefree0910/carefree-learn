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
from ....modules.blocks import Lambda


class MaskedConv2d(Conv2d):
    mask: Tensor

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
        need_embedding: bool = False,
        latent_channels: int = 128,
        *,
        num_layers: int = 6,
        norm_type: str = "batch",
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("`PixelCNN` requires `in_channels` to be 1")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_channels = latent_channels

        def _get_block(in_nc: int, out_nc: int, mask_type: str) -> nn.Sequential:
            return nn.Sequential(
                *get_conv_blocks(
                    in_nc,
                    out_nc,
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

        if not need_embedding:
            self.to_blocks = nn.Sequential(
                Lambda(lambda t: t.float() / (num_classes - 1), name="normalize"),
                _get_block(in_channels, latent_channels, "A"),
            )
        else:
            self.to_blocks = nn.Sequential(
                Lambda(lambda t: t.squeeze(1), name="squeeze"),
                nn.Embedding(num_classes, latent_channels),
                Lambda(lambda t: t.permute(0, 3, 1, 2), name="permute"),
                _get_block(latent_channels, latent_channels, "A"),
            )
        blocks: List[nn.Module] = []
        for i in range(num_layers - 1):
            blocks.append(_get_block(latent_channels, latent_channels, "B"))
        blocks.append(Conv2d(latent_channels, num_classes, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        net = self.to_blocks(net)
        return {PREDICTIONS_KEY: self.net(net)}

    def sample(self, num_sample: int, img_size: int, **kwargs: Any) -> Tensor:
        shape = num_sample, self.in_channels, img_size, img_size
        sampled = torch.zeros(shape, dtype=torch.long, device=self.device)
        for i in range(img_size):
            for j in range(img_size):
                out = self.forward(0, {INPUT_KEY: sampled}, **kwargs)[PREDICTIONS_KEY]
                probabilities = F.softmax(out[:, :, i, j], dim=1).data
                sampled[:, :, i, j] = torch.multinomial(probabilities, 1)
        return sampled


__all__ = ["PixelCNN"]
