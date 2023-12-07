import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Optional

from ..common import register_auto_regressor
from ..common import IAutoRegressor
from ...core import get_conv_blocks
from ...core import Conv2d
from ...core import ChannelPadding
from ...common import Lambda
from ....toolkit import get_device


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
        padding: Any = "same",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
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
        )
        self.register_buffer("mask", self.weight.data.clone())
        _, _, h, w = self.weight.shape
        self.mask.fill_(1.0)
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0.0
        self.mask[:, :, h // 2 + 1 :] = 0.0

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
    ) -> Tensor:
        self.weight.data *= self.mask
        return super().forward(net, style, transpose=transpose)


@register_auto_regressor("pixel_cnn")
class PixelCNN(IAutoRegressor):
    def __init__(
        self,
        num_codes: int,
        in_channels: int = 3,
        need_embedding: bool = False,
        latent_channels: int = 128,
        *,
        norm_type: Optional[str] = "batch",
        num_layers: int = 6,
        channel_padding: Optional[int] = 16,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("`PixelCNN` requires `in_channels` to be 1")
        self.in_channels = in_channels
        self.num_codes = num_codes
        self.latent_channels = latent_channels
        self.num_classes = num_classes

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

        # to blocks
        if not need_embedding:
            start_channels = in_channels
            normalize = lambda t: t.float() / (num_codes - 1)
            self.to_blocks = nn.Sequential(Lambda(normalize, name="normalize"))
        else:
            start_channels = latent_channels
            self.to_blocks = nn.Sequential(
                Lambda(lambda t: t.squeeze(1), name="squeeze"),
                nn.Embedding(num_codes, latent_channels),
                Lambda(lambda t: t.permute(0, 3, 1, 2), name="permute"),
            )
        # channel padding
        self.channel_padding = None
        if channel_padding is not None:
            self.channel_padding = ChannelPadding(
                start_channels,
                channel_padding,
                num_classes=num_classes,
            )
        elif num_classes is not None:
            msg = "`channel_padding` should be provided when `num_classes` is provided"
            raise ValueError(msg)
        # blocks
        blocks: List[nn.Module] = [_get_block(start_channels, latent_channels, "A")]
        for _ in range(num_layers - 1):
            blocks.append(_get_block(latent_channels, latent_channels, "B"))
        blocks.append(Conv2d(latent_channels, num_codes, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor, labels: Optional[Tensor]) -> Tensor:
        net = self.to_blocks(net)
        if self.channel_padding is not None:
            net = self.channel_padding(net, labels)
        return self.net(net)

    def sample(
        self,
        num_samples: int,
        *,
        img_size: int,
        labels: Optional[Tensor] = None,
        class_idx: Optional[int] = None,
    ) -> Tensor:
        device = get_device(self)
        shape = num_samples, self.in_channels, img_size, img_size
        sampled = torch.zeros(shape, dtype=torch.long, device=device)
        if not self.is_conditional:
            labels = None
        elif labels is None:
            if class_idx is not None:
                labels = self.get_sample_labels(num_samples, class_idx)
            else:
                labels = torch.randint(self.num_classes, [num_samples], device=device)
        for i in range(img_size):
            for j in range(img_size):
                out = self.forward(sampled, labels)
                probabilities = F.softmax(out[:, :, i, j], dim=1).data
                sampled[:, :, i, j] = torch.multinomial(probabilities, 1)
        return sampled


__all__ = [
    "PixelCNN",
]
