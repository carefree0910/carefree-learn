import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from cftool.misc import shallow_copy_dict

from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....misc.toolkit import WithRegister
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....modules.blocks import ImgToPatches


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


class Encoder1DFromPatches(Encoder1DBase, metaclass=ABCMeta):
    encoder: nn.Module

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(img_size, in_channels, latent_dim)
        self.to_patches = ImgToPatches(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            **(to_patches_configs or {}),
        )

    @property
    def num_patches(self) -> int:
        return self.to_patches.num_patches

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        inp = batch[INPUT_KEY]
        patches = self.to_patches(inp)
        batch[INPUT_KEY] = patches
        kwargs["hwp"] = *inp.shape[-2:], self.to_patches.patch_size
        return {LATENT_KEY: self.encoder(batch[INPUT_KEY], **kwargs)}


__all__ = [
    "EncoderBase",
    "Encoder1DBase",
    "Encoder1DFromPatches",
]
