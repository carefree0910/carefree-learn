import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import VanillaGANMixin
from ..implicit.siren import ImgSiren


@VanillaGANMixin.register("siren_gan")
class SirenGAN(VanillaGANMixin):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 128,
        num_classes: Optional[int] = None,
        conditional_dim: int = 16,
        *,
        num_layers: int = 4,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        bias: bool = True,
        final_activation: Optional[str] = None,
        discriminator: str = "basic",
        discriminator_configs: Optional[Dict[str, Any]] = None,
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
        self.latent_dim = latent_dim
        self.out_channels = out_channels or in_channels
        # siren
        self.siren = ImgSiren(
            img_size,
            self.out_channels,
            latent_dim,
            num_classes,
            conditional_dim,
            num_layers=num_layers,
            w_sin=w_sin,
            w_sin_initial=w_sin_initial,
            bias=bias,
            final_activation=final_activation,
        )

    def decode(
        self,
        z: Tensor,
        *,
        labels: Optional[Tensor],
        size: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.siren.decode(z, labels=labels, size=size)

    # training part

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.siren.parameters())


__all__ = ["SirenGAN"]
