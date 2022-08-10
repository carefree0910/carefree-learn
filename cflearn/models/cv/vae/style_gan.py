from typing import Any
from typing import Dict
from typing import Optional

from ..vae import VanillaVAE1D
from ....misc.internal_.register import register_module


@register_module("style_vae")
class StyleVAE(VanillaVAE1D):
    def __init__(
        self,
        img_size: int,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = None,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: int = 4,
        encoder: str = "vanilla",
        decoder: str = "style",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        if decoder not in ("style", "style2"):
            msg = f"`decoder` should be either 'style' or 'style2', '{decoder}' found"
            raise ValueError(msg)
        if decoder_config is None:
            decoder_config = {}
        v2 = decoder == "style2"
        if v2:
            decoder_config.setdefault("channel_base", img_size * 64)
        super().__init__(
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
            output_activation=None if v2 else "tanh",
        )


__all__ = [
    "StyleVAE",
]
