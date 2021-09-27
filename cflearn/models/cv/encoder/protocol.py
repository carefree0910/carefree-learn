import torch

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
from ....misc.toolkit import eval_context
from ....misc.toolkit import WithRegister
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....modules.blocks import ImgToPatches


encoders: Dict[str, Type["EncoderBase"]] = {}
encoders_1d: Dict[str, Type["Encoder1DBase"]] = {}


class EncoderProtocol(nn.Module):
    @abstractmethod
    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        pass

    def encode(self, batch: tensor_dict_type, **kwargs: Any) -> torch.Tensor:
        return self.forward(0, batch, **kwargs)[LATENT_KEY]


# encode to a latent feature map
class EncoderBase(EncoderProtocol, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["EncoderBase"]] = encoders

    def __init__(
        self,
        in_channels: int,
        num_downsample: int,
        latent_channels: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_downsample = num_downsample
        self.latent_channels = latent_channels

    def latent_resolution(self, img_size: int) -> int:
        shape = 1, self.in_channels, img_size, img_size
        params = list(self.parameters())
        device = "cpu" if not params else params[0].device
        with eval_context(self):
            net = self.encode({INPUT_KEY: torch.zeros(*shape, device=device)})
        return net.shape[2]


# encode to a 1d latent code
class Encoder1DBase(EncoderProtocol, WithRegister, metaclass=ABCMeta):
    d: Dict[str, Type["Encoder1DBase"]] = encoders_1d

    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim


class Encoder1DFromPatches(Encoder1DBase, metaclass=ABCMeta):
    encoder: nn.Module

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_channels, latent_dim)
        if to_patches_config is None:
            to_patches_config = {}
        to_patches_config.update(
            {
                "img_size": img_size,
                "patch_size": patch_size,
                "in_channels": in_channels,
                "latent_dim": latent_dim,
            }
        )
        self.to_patches = ImgToPatches.make(to_patches_type, to_patches_config)

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
        patches, hw = self.to_patches(inp)
        batch[INPUT_KEY] = patches
        kwargs["hw"] = hw
        kwargs["hwp"] = *inp.shape[-2:], self.to_patches.patch_size
        return {LATENT_KEY: self.encoder(batch[INPUT_KEY], **kwargs)}


__all__ = [
    "EncoderBase",
    "Encoder1DBase",
    "Encoder1DFromPatches",
]
