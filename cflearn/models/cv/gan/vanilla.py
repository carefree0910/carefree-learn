import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .schema import VanillaGANMixin
from ..decoder import make_decoder
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....misc.internal_.register import register_custom_module
from ....misc.internal_.register import CustomModule


@register_custom_module("gan")
class VanillaGAN(VanillaGANMixin, CustomModule):  # type: ignore
    generator: nn.Module

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
        self._initialize(
            in_channels=in_channels,
            discriminator=discriminator,
            discriminator_config=discriminator_config,
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_config=gan_loss_config,
        )
        self.latent_dim = latent_dim
        # generator
        if generator_config is None:
            generator_config = {}
        generator_config["img_size"] = img_size
        generator_config["latent_dim"] = latent_dim
        generator_config["latent_resolution"] = latent_resolution
        generator_config["out_channels"] = out_channels or in_channels
        generator_config["num_classes"] = num_classes
        self.generator = make_decoder(generator, generator_config, is_1d=True)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        batch = {INPUT_KEY: z, LABEL_KEY: labels}
        net = self.generator.decode(batch, **kwargs)
        return net


__all__ = ["VanillaGAN"]
