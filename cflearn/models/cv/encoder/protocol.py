import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional

from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import WithRegister


encoders: Dict[str, Type["EncoderBase"]] = {}


class EncoderBase(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["EncoderBase"]] = encoders

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_downsample: int,
        latent_channels: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_downsample = num_downsample
        self.latent_channels = latent_channels

    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def encode(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        return self.forward(0, batch, **kwargs)


__all__ = ["EncoderBase"]
