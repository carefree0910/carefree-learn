import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..decoder import StyleGANDecoder
from ..encoder import Encoder1DBase
from ..encoder import Encoder1DFromPatches
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import ModelProtocol
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ..vae.losses import VAELoss
from ..vae.vanilla import reparameterize
from ..vae.constants import MU_KEY
from ..vae.constants import LOG_VAR_KEY


@ModelProtocol.register("style_vae")
class StyleVAE(ModelProtocol, GaussianGeneratorMixin):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 128,
        in_channels: int = 3,
        *,
        # encoder
        encoder1d: str = "vanilla",
        encoder1d_config: Optional[Dict[str, Any]] = None,
        num_downsample: int = 4,
        # decoder
        first_channels: int = 64,
        channel_max: int = 256,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        block_kwargs: Optional[Dict[str, Any]] = None,
        # loss
        vae_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_classes = num_classes
        if encoder1d_config is None:
            encoder1d_config = {}
        if Encoder1DFromPatches.check_subclass(encoder1d):
            encoder1d_config["img_size"] = img_size
        encoder1d_config["in_channels"] = in_channels
        encoder1d_config["latent_dim"] = latent_dim * 2
        encoder1d_config["num_downsample"] = num_downsample
        self.encoder = Encoder1DBase.make(encoder1d, encoder1d_config)
        self.decoder = StyleGANDecoder(
            img_size,
            latent_dim,
            self.out_channels,
            channel_base=img_size * first_channels,
            channel_max=channel_max,
            num_classes=num_classes,
            conv_clamp=conv_clamp,
            **(block_kwargs or {}),
        )
        self.vae_loss = VAELoss(**(vae_loss_config or {}))

    @property
    def can_reconstruct(self) -> bool:
        return True

    def _z2ws(self, z: Tensor) -> Tensor:
        return z.unsqueeze(1).repeat(1, self.decoder.num_ws, 1)

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        net = self.decoder(self._z2ws(z), labels=labels, **kwargs)
        return net

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.encoder.encode(batch)
        z_mu, z_log_var = net.chunk(2, dim=1)
        z = reparameterize(z_mu, z_log_var)
        net = self.decode(z, labels=batch.get(LABEL_KEY), **kwargs)
        return {PREDICTIONS_KEY: net, MU_KEY: z_mu, LOG_VAR_KEY: z_log_var}


__all__ = [
    "StyleVAE",
]
