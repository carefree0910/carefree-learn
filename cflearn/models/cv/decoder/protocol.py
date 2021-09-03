import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..toolkit import auto_num_layers
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....misc.toolkit import interpolate
from ....misc.toolkit import WithRegister
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....modules.blocks import ChannelPadding


decoders: Dict[str, Type["DecoderBase"]] = {}


class DecoderBase(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DecoderBase"]] = decoders

    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        *,
        img_size: Optional[int] = None,
        num_upsample: Optional[int] = None,
        cond_channels: int = 16,
        num_classes: Optional[int] = None,
        latent_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_channels = latent_channels
        self.latent_resolution = latent_resolution
        self.out_channels = out_channels
        if num_upsample is None:
            fmt = "`{}` should be provided when `num_upsample` is not"
            if img_size is None:
                raise ValueError(fmt.format("img_size"))
            if latent_resolution is None:
                raise ValueError(fmt.format("latent_resolution"))
            num_upsample = auto_num_layers(img_size, latent_resolution, None)
        self.num_upsample = num_upsample
        # conditional
        self.cond_channels = cond_channels
        self.num_classes = num_classes
        if num_classes is None:
            self.cond = None
        else:
            if latent_resolution is None:
                msg = "`latent_resolution` should be provided for conditional modeling"
                raise ValueError(msg)
            self.cond = ChannelPadding(
                cond_channels,
                latent_resolution,
                num_classes=num_classes,
            )

    @property
    def is_conditional(self) -> bool:
        return self.num_classes is not None

    def _inject_cond(self, batch: tensor_dict_type) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        if self.cond is not None:
            batch[INPUT_KEY] = self.cond(batch[INPUT_KEY], batch[LABEL_KEY])
        return batch

    def resize(self, net: Tensor) -> Tensor:
        if self.img_size is None:
            return net
        return interpolate(net, size=self.img_size)

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def decode(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        return self.forward(0, batch, **kwargs)


__all__ = ["DecoderBase"]
