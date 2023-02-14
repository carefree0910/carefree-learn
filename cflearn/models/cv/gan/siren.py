import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .schema import VanillaGANMixin
from ...implicit.siren import ImgSiren
from ....register import register_custom_module
from ....register import CustomModule


@register_custom_module("siren_gan")
class SirenGAN(VanillaGANMixin, CustomModule):  # type: ignore
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
        discriminator_config: Optional[Dict[str, Any]] = None,
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

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.siren.parameters())

    def decode(
        self,
        z: Tensor,
        *,
        labels: Optional[Tensor],
        size: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        return self.siren.decode(z, labels=labels, size=size)


__all__ = ["SirenGAN"]
