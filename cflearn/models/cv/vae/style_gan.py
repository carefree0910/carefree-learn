import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..encoder import Encoder1DBase
from ..encoder import Encoder1DFromPatches
from ..protocol import GaussianGeneratorMixin
from ..generator import StyleGANGenerator
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
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
        latent_dim: int = 256,
        in_channels: int = 3,
        *,
        encoder1d: str = "vanilla",
        encoder1d_config: Optional[Dict[str, Any]] = None,
        num_downsample: int = 4,
        num_layers: int = 2,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        block_kwargs: Optional[Dict[str, Any]] = None,
        mapping_kwargs: Optional[Dict[str, Any]] = None,
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
        self.decoder = StyleGANGenerator(
            img_size,
            latent_dim,
            self.out_channels,
            num_layers=num_layers,
            channel_base=channel_base,
            channel_max=channel_max,
            num_content_classes=num_classes,
            conv_clamp=conv_clamp,
            block_kwargs=block_kwargs,
            mapping_kwargs=mapping_kwargs,
        )
        self.vae_loss = VAELoss(**(vae_loss_config or {}))

    @property
    def can_reconstruct(self) -> bool:
        return True

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        rs = self.decoder(0, {INPUT_KEY: z, LABEL_KEY: labels}, **kwargs)
        net = torch.tanh(rs[PREDICTIONS_KEY])
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
        decoder_batch = {INPUT_KEY: z}
        content_labels = batch.get(LABEL_KEY)
        if content_labels is not None:
            decoder_batch[LABEL_KEY] = content_labels
        net = self.decoder(batch_idx, decoder_batch, state, **kwargs)[PREDICTIONS_KEY]
        net = torch.tanh(net)
        return {PREDICTIONS_KEY: net, MU_KEY: z_mu, LOG_VAR_KEY: z_log_var}


__all__ = [
    "StyleVAE",
]
