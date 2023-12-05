from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..common import build_decoder
from ..common import register_generator
from ..common import build_discriminator
from ..common import DecoderInputs
from ..common import IGaussianGenerator


@register_generator("gan")
class GAN(IGaussianGenerator):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 64,
        latent_resolution: int = 7,
        *,
        generator: str = "vanilla_1d",
        discriminator: str = "basic",
        generator_config: Optional[Dict[str, Any]] = None,
        discriminator_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # generator
        if generator_config is None:
            generator_config = {}
        generator_config["img_size"] = img_size
        generator_config["latent_dim"] = latent_dim
        generator_config["latent_resolution"] = latent_resolution
        generator_config["out_channels"] = out_channels or in_channels
        generator_config["num_classes"] = num_classes
        self.generator = build_decoder(generator, config=generator_config)
        # discriminator
        if discriminator_config is None:
            discriminator_config = {}
        discriminator_config["in_channels"] = in_channels
        discriminator_config["num_classes"] = num_classes
        self.discriminator = build_discriminator(
            discriminator,
            config=discriminator_config,
        )

    def forward(self, inputs: DecoderInputs) -> Tensor:
        return self.generator.decode(inputs)


__all__ = [
    "GAN",
]
