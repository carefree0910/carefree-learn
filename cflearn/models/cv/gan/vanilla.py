from typing import Any
from typing import Dict
from typing import Optional

from .schema import IVanillaGAN
from .discriminators import DiscriminatorBase
from ..decoder import make_decoder


@IVanillaGAN.register("gan")
class VanillaGAN(IVanillaGAN):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 64,
        latent_resolution: int = 7,
        *,
        generator: str = "vanilla",
        discriminator: str = "basic",
        generator_config: Optional[Dict[str, Any]] = None,
        discriminator_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.build_loss(
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_config=gan_loss_config,
        )
        # generator
        if generator_config is None:
            generator_config = {}
        generator_config["img_size"] = img_size
        generator_config["latent_dim"] = latent_dim
        generator_config["latent_resolution"] = latent_resolution
        generator_config["out_channels"] = out_channels or in_channels
        generator_config["num_classes"] = num_classes
        self.generator = make_decoder(generator, generator_config, is_1d=True)  # type: ignore
        # discriminator
        if discriminator_config is None:
            discriminator_config = {}
        discriminator_config["in_channels"] = in_channels
        discriminator_config["num_classes"] = num_classes
        self.discriminator = DiscriminatorBase.make(
            discriminator,
            config=discriminator_config,
        )


__all__ = ["VanillaGAN"]
