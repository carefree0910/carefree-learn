import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import VanillaGANMixin
from ..decoder import DecoderMixin
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....misc.toolkit import auto_num_layers
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
        latent_resolution: int = 2,
        *,
        generator: str = "vanilla",
        discriminator: str = "basic",
        generator_config: Optional[Dict[str, Any]] = None,
        discriminator_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            in_channels,
            discriminator=discriminator,
            discriminator_config=discriminator_config,
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_config=gan_loss_config,
        )
        num_upsample = auto_num_layers(
            img_size,
            latent_resolution,
            None,
            use_stride=True,
        )
        # latent
        self.latent_dim = latent_dim
        map_area = latent_resolution**2
        if latent_dim % map_area != 0:
            msg = f"`latent_dim` should be divisible by `map_area` ({map_area})"
            raise ValueError(msg)
        compressed_channels = latent_dim // map_area
        shape = -1, compressed_channels, latent_resolution, latent_resolution
        self.from_latent = nn.Sequential(
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(compressed_channels, latent_dim, kernel_size=1, bias=False),
        )
        # generator
        if generator_config is None:
            generator_config = {}
        generator_config["img_size"] = img_size
        generator_config["latent_channels"] = latent_dim
        generator_config["latent_resolution"] = latent_resolution
        generator_config["num_upsample"] = num_upsample
        generator_config["out_channels"] = out_channels or in_channels
        generator_config["num_classes"] = num_classes
        self.generator = DecoderMixin.make(generator, config=generator_config)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        batch = {INPUT_KEY: self.from_latent(z), LABEL_KEY: labels}
        net = self.generator.decode(batch, **kwargs)
        return net


__all__ = ["VanillaGAN"]
