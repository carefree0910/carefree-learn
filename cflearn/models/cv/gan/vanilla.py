import math
import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import VanillaGANMixin
from ..decoder import DecoderBase
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda


@VanillaGANMixin.register("gan")
class VanillaGAN(VanillaGANMixin):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 128,
        latent_channels: int = 128,
        latent_resolution: int = 2,
        *,
        generator: str = "vanilla",
        discriminator: str = "basic",
        generator_configs: Optional[Dict[str, Any]] = None,
        discriminator_configs: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            img_size,
            in_channels,
            discriminator=discriminator,
            discriminator_configs=discriminator_configs,
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_configs=gan_loss_configs,
        )
        num_upsample = math.ceil(math.log2(img_size / latent_resolution))
        # latent
        self.latent_dim = latent_dim
        map_area = latent_resolution ** 2
        if latent_dim % map_area != 0:
            msg = f"`latent_dim` should be divisible by `map_area` ({map_area})"
            raise ValueError(msg)
        compressed_channels = latent_dim // map_area
        shape = -1, compressed_channels, latent_resolution, latent_resolution
        self.from_latent = nn.Sequential(
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(compressed_channels, latent_channels, kernel_size=1, bias=False),
        )
        # generator
        if generator_configs is None:
            generator_configs = {}
        generator_configs["img_size"] = img_size
        generator_configs["latent_channels"] = latent_channels
        generator_configs["latent_resolution"] = latent_resolution
        generator_configs["num_upsample"] = num_upsample
        generator_configs["out_channels"] = out_channels or in_channels
        generator_configs["num_classes"] = num_classes
        self.generator = DecoderBase.make(generator, config=generator_configs)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        batch = {INPUT_KEY: self.from_latent(z), LABEL_KEY: labels}
        net = self.generator.decode(batch, **kwargs)[PREDICTIONS_KEY]
        return torch.tanh(net)


__all__ = ["VanillaGAN"]
