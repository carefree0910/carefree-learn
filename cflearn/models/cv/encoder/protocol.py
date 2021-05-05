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
encoders_1d: Dict[str, Type["Encoder1DBase"]] = {}


# encode to a latent feature map
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


# encode to a 1d latent code
class Encoder1DBase(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["Encoder1DBase"]] = encoders_1d

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim

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


__all__ = [
    "EncoderBase",
    "Encoder1DBase",
]
