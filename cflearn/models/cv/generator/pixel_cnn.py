import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from ....register import register_module
from ....constants import INPUT_KEY
from ....constants import ORIGINAL_LABEL_KEY
from ...schemas.cv import ImageTranslatorMixin
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import ChannelPadding


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


@register_module("pixel_cnn")
class PixelCNN(nn.Module, ImageTranslatorMixin):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        need_embedding: bool = False,
        latent_channels: int = 128,
        *,
        norm_type: Optional[str] = "batch",
        num_layers: int = 6,
        channel_padding: Optional[int] = 16,
        num_conditional_classes: Optional[int] = None,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("`PixelCNN` requires `in_channels` to be 1")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.num_conditional_classes = num_conditional_classes

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
            normalize = lambda t: t.float() / (num_classes - 1)
            self.to_blocks = nn.Sequential(Lambda(normalize, name="normalize"))
        else:
            start_channels = latent_channels
            self.to_blocks = nn.Sequential(
                Lambda(lambda t: t.squeeze(1), name="squeeze"),
                nn.Embedding(num_classes, latent_channels),
                Lambda(lambda t: t.permute(0, 3, 1, 2), name="permute"),
            )
        # channel padding
        self.channel_padding = None
        if channel_padding is not None:
            self.channel_padding = ChannelPadding(
                start_channels,
                channel_padding,
                num_classes=num_conditional_classes,
            )
        elif num_conditional_classes is not None:
            raise ValueError(
                "`channel_padding` should be provided "
                "when `num_conditional_classes` is provided"
            )
        # blocks
        blocks: List[nn.Module] = [_get_block(start_channels, latent_channels, "A")]
        for i in range(num_layers - 1):
            blocks.append(_get_block(latent_channels, latent_channels, "B"))
        blocks.append(Conv2d(latent_channels, num_classes, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, batch: tensor_dict_type) -> Tensor:
        net = batch[INPUT_KEY]
        net = self.to_blocks(net)
        if self.channel_padding is not None:
            net = self.channel_padding(net, batch[ORIGINAL_LABEL_KEY])
        return self.net(net)

    def sample(
        self,
        num_sample: int,
        img_size: int,
        class_idx: Optional[int] = None,
    ) -> Tensor:
        shape = num_sample, self.in_channels, img_size, img_size
        sampled = torch.zeros(shape, dtype=torch.long, device=self.device)
        if self.num_conditional_classes is None:
            labels = None
        else:
            if class_idx is not None:
                labels = torch.full([num_sample], class_idx, device=self.device)
            else:
                labels = torch.randint(
                    self.num_conditional_classes,
                    [num_sample],
                    device=self.device,
                )
        for i in range(img_size):
            for j in range(img_size):
                batch = {INPUT_KEY: sampled, ORIGINAL_LABEL_KEY: labels}
                out = self.forward(batch)
                probabilities = F.softmax(out[:, :, i, j], dim=1).data
                sampled[:, :, i, j] = torch.multinomial(probabilities, 1)
        return sampled


__all__ = ["PixelCNN"]
