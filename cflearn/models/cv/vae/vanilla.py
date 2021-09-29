import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from .constants import MU_KEY
from .constants import LOG_VAR_KEY
from ..general import EncoderDecoder
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Lambda
from ....modules.blocks import Activations
from ....modules.blocks import ChannelPadding


def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


class VanillaVAEBase(ModelProtocol, GaussianGeneratorMixin):
    from_latent: nn.Module

    def __init__(
        self,
        is_id: bool,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = 4,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        output_activation: Optional[str] = "tanh",
    ):
        super().__init__()
        # encoder defaults
        if encoder_config is None:
            encoder_config = {}
        if encoder == "vanilla":
            encoder_config.setdefault("norm_type", "instance")
            encoder_config.setdefault("padding", "reflection")
        # decoder defaults
        if decoder_config is None:
            decoder_config = {}
        if decoder == "vanilla":
            decoder_config.setdefault("last_kernel_size", 7)
            decoder_config.setdefault("reduce_channel_on_upsample", True)
        # build
        self.generator = EncoderDecoder(
            is_id,
            in_channels,
            out_channels,
            target_downsample,
            latent_padding_channels,
            num_classes,
            latent=latent * 2,
            decoder_latent=latent,
            img_size=img_size,
            min_size=min_size,
            num_downsample=num_downsample,
            num_upsample=num_upsample,
            latent_resolution=latent_resolution,
            encoder=encoder,
            decoder=decoder,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        self.latent_dim = latent
        self.num_classes = num_classes
        if output_activation is None:
            self.out = None
        else:
            self.out = Activations.make(output_activation)

    @property
    def can_reconstruct(self) -> bool:
        return True

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        batch = {INPUT_KEY: self.from_latent(z), LABEL_KEY: labels}
        net = self.generator.decode(batch, **kwargs)
        if self.out is not None:
            net = self.out(net)
        return net

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.generator.encode(batch, **kwargs)
        mu, log_var = net.chunk(2, dim=1)
        net = reparameterize(mu, log_var)
        labels = None if self.num_classes is None else batch[LABEL_KEY].view(-1)
        net = self.decode(net, labels=labels, **kwargs)
        return {PREDICTIONS_KEY: net, MU_KEY: mu, LOG_VAR_KEY: log_var}


@ModelProtocol.register("vae")
@ModelProtocol.register("vae1d")
class VanillaVAE1D(VanillaVAEBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = 4,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        output_activation: Optional[str] = "tanh",
    ):
        if decoder_config is None:
            decoder_config = {}
        if decoder == "vanilla":
            decoder_config["latent_expand_ratio"] = 2
        super().__init__(
            True,
            in_channels,
            out_channels,
            target_downsample,
            latent_padding_channels,
            num_classes,
            latent=latent,
            img_size=img_size,
            min_size=min_size,
            num_downsample=num_downsample,
            num_upsample=num_upsample,
            latent_resolution=latent_resolution,
            encoder=encoder,
            decoder=decoder,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            output_activation=output_activation,
        )
        self.from_latent = nn.Identity()


@ModelProtocol.register("vae2d")
class VanillaVAE2D(VanillaVAEBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = 4,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        output_activation: Optional[str] = "tanh",
    ):
        super().__init__(
            False,
            in_channels,
            out_channels,
            target_downsample,
            latent_padding_channels,
            num_classes,
            latent=latent,
            img_size=img_size,
            min_size=min_size,
            num_downsample=num_downsample,
            num_upsample=num_upsample,
            latent_resolution=latent_resolution,
            encoder=encoder,
            decoder=decoder,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            output_activation=output_activation,
        )
        self.latent_resolution = self.generator.latent_resolution
        assert self.latent_resolution is not None
        self.latent_dim = latent * self.latent_resolution ** 2
        shape = -1, latent, self.latent_resolution, self.latent_resolution
        blocks = [Lambda(lambda net: net.view(shape), f"reshape -> {shape}")]
        if latent_padding_channels is None:
            self.from_latent = blocks[0]
        else:
            blocks.append(
                ChannelPadding(
                    latent,
                    latent_padding_channels,
                    self.latent_resolution,
                )
            )
            self.from_latent = nn.Sequential(*blocks)


__all__ = [
    "VanillaVAE1D",
    "VanillaVAE2D",
]
