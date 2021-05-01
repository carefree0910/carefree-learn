import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import WithRegister


decoders: Dict[str, Type["DecoderBase"]] = {}


class DecoderBase(nn.Module, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["DecoderBase"]] = decoders

    def __init__(
        self,
        img_size: int,
        latent_channels: int,
        out_channels: int,
        last_kernel_size: int = 7,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.last_kernel_size = last_kernel_size

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
