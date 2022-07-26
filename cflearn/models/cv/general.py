import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from .encoder import make_encoder
from .encoder import EncoderMixin
from .decoder import make_decoder
from ...types import tensor_dict_type
from ...misc.toolkit import auto_num_layers


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        is_1d: bool,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = None,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        decoder_latent: Optional[int] = None,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # up / down sample stuffs
        if num_downsample is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `num_downsample` should be provided "
                    "(when `num_downsample` is not provided, it will be inferred "
                    "automatically with `img_size` & `min_size`)"
                )
            args = img_size, min_size, target_downsample
            num_downsample = auto_num_layers(*args, use_stride=encoder == "vanilla")
        num_upsample = num_upsample or num_downsample
        if latent_resolution is None and img_size is None:
            raise ValueError(
                "either `img_size` or `latent_resolution` should be provided "
                "in `VanillaVAEBase` (when `latent_resolution` is not provided, "
                "it will be inferred automatically with `img_size`)"
            )
        # properties
        self.is_1d = is_1d
        self.latent = latent
        self.decoder_latent = decoder_latent or latent
        self.latent_key = "latent_dim" if is_1d else "latent_channels"
        self.img_size = img_size
        self.latent_padding_channels = latent_padding_channels
        self.num_classes = num_classes
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample
        # encoder
        if encoder_config is None:
            encoder_config = {}
        encoder_config.setdefault("img_size", img_size)
        encoder_config.setdefault("in_channels", in_channels)
        encoder_config.setdefault("num_downsample", num_downsample)
        encoder_config[self.latent_key] = latent
        self.encoder = make_encoder(encoder, encoder_config, is_1d=is_1d)  # type: ignore
        if isinstance(self.encoder, EncoderMixin) and img_size is not None:
            latent_resolution = self.encoder.latent_resolution(img_size)
        # decoder
        if decoder_config is None:
            decoder_config = {}
        decoder_config.setdefault("img_size", img_size)
        decoder_config.setdefault("num_classes", num_classes)
        decoder_config.setdefault("out_channels", out_channels or in_channels)
        decoder_config.setdefault("num_upsample", num_upsample)
        decoder_config.setdefault("latent_resolution", latent_resolution)
        decoder_config[self.latent_key] = self.decoder_latent
        self.decoder = make_decoder(decoder, decoder_config, is_1d=is_1d)
        self.latent_resolution = self.decoder.latent_resolution

    def resize(self, net: Tensor) -> Tensor:
        return self.decoder.resize(net)

    def encode(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        return self.encoder.encode(batch, **kwargs)

    def decode(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        net = self.decoder.decode(batch, **kwargs)
        return self.resize(net)


__all__ = ["EncoderDecoder"]
