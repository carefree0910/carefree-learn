import math
import torch

import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Optional

from .constants import MU_KEY
from .constants import LOG_VAR_KEY
from ..encoder import EncoderBase
from ..encoder import Encoder1DBase
from ..decoder import DecoderBase
from ..toolkit import get_latent_resolution
from ..toolkit import auto_num_layers
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import INFO_PREFIX
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import interpolate
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Linear
from ....modules.blocks import ChannelPadding


def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


class VanillaVAEBase(ModelProtocol, GaussianGeneratorMixin):
    encoder_base: Union[Type[EncoderBase], Type[Encoder1DBase]]
    to_statistics: nn.Module
    from_latent: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: int = 4,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        # up / down sample stuffs
        num_upsample = num_upsample or num_downsample
        if num_downsample is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `num_downsample` should be provided "
                    "(when `num_downsample` is not provided, it will be inferred "
                    "automatically with `img_size` & `min_size`)"
                )
            num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        if latent_resolution is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `latent_resolution` should be provided "
                    "(when `latent_resolution` is not provided, it will be inferred "
                    "automatically with `img_size`)"
                )
            latent_resolution = get_latent_resolution(img_size, num_downsample)
        if img_size is None:
            raw_size = latent_resolution * 2 ** num_downsample
            print(f"{INFO_PREFIX}img_size is not provided, raw_size will be {raw_size}")
        # properties
        latent_d = kwargs.get(self.key, 128)
        self.latent_d = latent_d
        self.latent_padding_channels = latent_padding_channels
        self.num_classes = num_classes
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample
        self.latent_resolution = latent_resolution
        # build
        self._build(
            in_channels,
            out_channels,
            img_size,
            encoder,
            decoder,
            encoder_config,
            decoder_config,
        )

    @property
    def key(self) -> str:
        return "latent_dim" if self.is_1d else "latent_channels"

    @property
    def is_1d(self) -> bool:
        return self.encoder_base is Encoder1DBase

    @property
    def can_reconstruct(self) -> bool:
        return True

    @abstractmethod
    def _build(
        self,
        in_channels: int,
        out_channels: Optional[int],
        img_size: Optional[int],
        encoder: str,
        decoder: str,
        encoder_config: Optional[Dict[str, Any]],
        decoder_config: Optional[Dict[str, Any]],
    ) -> None:
        pass

    def _encoder(
        self,
        latent_d: int,
        img_size: Optional[int],
        in_channels: int,
        num_downsample: int,
        encoder: str,
        encoder_config: Optional[Dict[str, Any]],
    ) -> None:
        if encoder_config is None:
            encoder_config = {}
        if encoder != "backbone":
            encoder_config["img_size"] = img_size
        encoder_config["in_channels"] = in_channels
        encoder_config[self.key] = latent_d
        encoder_config["num_downsample"] = num_downsample
        self.encoder = self.encoder_base.make(encoder, config=encoder_config)

    def _decoder(
        self,
        latent_d: int,
        img_size: Optional[int],
        in_channels: int,
        latent_resolution: int,
        num_classes: Optional[int],
        out_channels: Optional[int],
        decoder: str,
        decoder_config: Optional[Dict[str, Any]],
    ) -> None:
        if decoder_config is None:
            decoder_config = {}
        decoder_config["img_size"] = img_size
        decoder_config["latent_channels"] = latent_d
        decoder_config["latent_resolution"] = latent_resolution
        decoder_config["num_upsample"] = self.num_upsample
        decoder_config["out_channels"] = out_channels or in_channels
        decoder_config["num_classes"] = num_classes
        self.decoder = DecoderBase.make(decoder, config=decoder_config)

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        batch = {INPUT_KEY: self.from_latent(z), LABEL_KEY: labels}
        net = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        inp = batch[INPUT_KEY]
        net = self.encoder.encode(batch, **kwargs)[LATENT_KEY]
        net = self.to_statistics(net)
        mu, log_var = net.chunk(2, dim=1)
        net = reparameterize(mu, log_var)
        labels = None if self.num_classes is None else batch[LABEL_KEY].view(-1)
        net = self.decode(net, labels=labels, **kwargs)
        net = interpolate(net, anchor=inp)
        return {PREDICTIONS_KEY: net, MU_KEY: mu, LOG_VAR_KEY: log_var}


@ModelProtocol.register("vae")
@ModelProtocol.register("vae1d")
class VanillaVAE1D(VanillaVAEBase):
    encoder_base = Encoder1DBase

    def _build(
        self,
        in_channels: int,
        out_channels: Optional[int],
        img_size: Optional[int],
        encoder: str,
        decoder: str,
        encoder_config: Optional[Dict[str, Any]],
        decoder_config: Optional[Dict[str, Any]],
    ) -> None:
        latent_d = self.latent_dim = self.latent_d
        self._encoder(
            latent_d,
            img_size,
            in_channels,
            self.num_downsample,
            encoder,
            encoder_config,
        )
        self.to_statistics = Linear(latent_d, 2 * latent_d, bias=False)
        latent_resolution = self.latent_resolution
        latent_area = latent_resolution ** 2
        latent_channels = math.ceil(self.latent_d / latent_area)
        shape = -1, latent_channels, latent_resolution, latent_resolution
        blocks: List[nn.Module] = [
            Linear(self.latent_d, latent_channels * latent_area),
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(latent_channels, self.latent_d, kernel_size=1, bias=False),
        ]
        lpc = self.latent_padding_channels
        if lpc is not None:
            latent_d += lpc
            latent_padding = ChannelPadding(lpc, latent_resolution)
            blocks.append(latent_padding)
        self.from_latent = nn.Sequential(*blocks)
        self._decoder(
            latent_d,
            img_size,
            in_channels,
            latent_resolution,
            self.num_classes,
            out_channels,
            decoder,
            decoder_config,
        )


@ModelProtocol.register("vae2d")
class VanillaVAE2D(VanillaVAEBase):
    encoder_base = EncoderBase

    def _build(
        self,
        in_channels: int,
        out_channels: Optional[int],
        img_size: Optional[int],
        encoder: str,
        decoder: str,
        encoder_config: Optional[Dict[str, Any]],
        decoder_config: Optional[Dict[str, Any]],
    ) -> None:
        latent_d = self.latent_d
        self._encoder(
            latent_d,
            img_size,
            in_channels,
            self.num_downsample,
            encoder,
            encoder_config,
        )
        self.latent_dim = latent_d * self.latent_resolution ** 2
        self.to_statistics = Conv2d(
            latent_d,
            latent_d * 2,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        latent_resolution = self.latent_resolution
        shape = -1, latent_d, latent_resolution, latent_resolution
        blocks = [Lambda(lambda net: net.view(shape), f"reshape -> {shape}")]
        lpc = self.latent_padding_channels
        if lpc is None:
            self.from_latent = blocks[0]
        else:
            latent_d += lpc
            latent_padding = ChannelPadding(lpc, latent_resolution)
            blocks.append(latent_padding)
            self.from_latent = nn.Sequential(*blocks)
        self._decoder(
            latent_d,
            img_size,
            in_channels,
            latent_resolution,
            self.num_classes,
            out_channels,
            decoder,
            decoder_config,
        )


__all__ = [
    "VanillaVAE1D",
    "VanillaVAE2D",
]
