import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister

from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....misc.toolkit import _forward
from ....misc.toolkit import filter_kw
from ....misc.toolkit import eval_context
from ....modules.blocks import ImgToPatches


encoders: Dict[str, Type["EncoderMixin"]] = {}
encoders_1d: Dict[str, Type["Encoder1DMixin"]] = {}


class IEncoder:
    def encode(self, batch: tensor_dict_type, **kwargs: Any) -> torch.Tensor:
        return run_encoder(self, 0, batch, **kwargs)[LATENT_KEY]


# encode to a latent feature map
class EncoderMixin(IEncoder, WithRegister["EncoderMixin"]):
    d = encoders

    in_channels: int
    num_downsample: int
    latent_channels: int

    def latent_resolution(self, img_size: int) -> int:
        shape = 1, self.in_channels, img_size, img_size
        params = list(self.parameters())
        device = "cpu" if not params else params[0].device
        with eval_context(self):
            net = self.encode({INPUT_KEY: torch.zeros(*shape, device=device)})
        return net.shape[2]


# encode to a 1d latent code
class Encoder1DMixin(IEncoder, WithRegister["Encoder1DMixin"]):
    d = encoders_1d

    in_channels: int
    latent_dim: int


def make_encoder(name: str, config: Dict[str, Any], *, is_1d: bool = False) -> IEncoder:
    base = (Encoder1DMixin if is_1d else EncoderMixin).get(name)  # type: ignore
    return base(**filter_kw(base, config))


def run_encoder(
    encoder: IEncoder,
    batch_idx: int,
    batch: tensor_dict_type,
    state: Optional[TrainerState] = None,
    **kwargs: Any,
) -> tensor_dict_type:
    return _forward(
        encoder,
        batch_idx,
        batch,
        INPUT_KEY,
        state,
        general_output_key=LATENT_KEY,
        **kwargs,
    )


# from patches
class EncoderFromPatchesMixin:
    encoder: nn.Module

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
    ):
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

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> torch.Tensor:
        batch = shallow_copy_dict(batch)
        inp = batch[INPUT_KEY]
        determinate = kwargs.pop("determinate", False)
        patches, hw = self.to_patches(inp, determinate=determinate)
        batch[INPUT_KEY] = patches
        kwargs["hw"] = hw
        kwargs["hwp"] = *inp.shape[-2:], self.to_patches.patch_size
        if check_requires(self.encoder.forward, "determinate", strict=False):
            kwargs["determinate"] = determinate
        return self.encoder(batch[INPUT_KEY], **kwargs)


class Encoder1DFromPatches(EncoderFromPatchesMixin, Encoder1DMixin):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        self.in_channels = in_channels
        self.latent_dim = latent_dim


class Encoder2DFromPatches(EncoderFromPatchesMixin, EncoderMixin):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_channels: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_channels,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        self.in_channels = in_channels
        self.latent_channels = latent_channels


__all__ = [
    "make_encoder",
    "run_encoder",
    "EncoderMixin",
    "Encoder1DMixin",
    "Encoder1DFromPatches",
    "Encoder2DFromPatches",
]
