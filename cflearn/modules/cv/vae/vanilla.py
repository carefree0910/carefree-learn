import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.types import tensor_dict_type

from .losses import MU_KEY
from .losses import LOG_VAR_KEY
from ..common import build_encoder
from ..common import build_decoder
from ..common import register_generator
from ..common import DecoderInputs
from ..common import IGaussianGenerator
from ....constants import PREDICTIONS_KEY


def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


@register_generator("vae")
class VAE(IGaussianGenerator):
    def __init__(
        self,
        in_channels: int,
        num_downsample: int,
        out_channels: Optional[int] = None,
        *,
        latent: int = 128,
        encoder: str = "vanilla_1d",
        decoder: str = "vanilla_1d",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        apply_tanh: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent
        self.num_classes = num_classes
        # encoder
        if encoder_config is None:
            encoder_config = {}
        if encoder.startswith("vanilla"):
            encoder_config.setdefault("norm_type", "instance")
            encoder_config.setdefault("padding", "reflection")
        encoder_config["in_channels"] = in_channels
        encoder_config["num_downsample"] = num_downsample
        encoder_config["latent_dim"] = latent * 2
        self.encoder = build_encoder(encoder, config=encoder_config)
        # decoder
        if decoder_config is None:
            decoder_config = {}
        if decoder.startswith("vanilla"):
            decoder_config.setdefault("last_kernel_size", 7)
            decoder_config.setdefault("reduce_channel_on_upsample", True)
        decoder_config["out_channels"] = out_channels or in_channels
        decoder_config["num_upsample"] = num_downsample
        decoder_config["latent_dim"] = latent
        decoder_config["num_classes"] = num_classes
        decoder_config["apply_tanh"] = apply_tanh
        self.decoder = build_decoder(decoder, config=decoder_config)

    def decode(self, inputs: DecoderInputs) -> Tensor:
        z = inputs.z
        if inputs.labels is None and self.num_classes is not None:
            shape = z.shape[0], 1
            inputs.labels = torch.randint(self.num_classes, shape, device=z.device)
        net = self.decoder.decode(inputs)
        return net

    def forward(self, net: Tensor, labels: Optional[Tensor] = None) -> tensor_dict_type:
        net = self.encoder.encode(net)
        mu, log_var = net.chunk(2, dim=1)
        z = reparameterize(mu, log_var)
        if self.num_classes is None:
            labels = None
        net = self.decode(DecoderInputs(z=z, labels=labels))
        return {PREDICTIONS_KEY: net, MU_KEY: mu, LOG_VAR_KEY: log_var}

    def reconstruct(
        self,
        net: Tensor,
        *,
        labels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        return self(net, labels)[PREDICTIONS_KEY]


__all__ = [
    "reparameterize",
    "VAE",
]
