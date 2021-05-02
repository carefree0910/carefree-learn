import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..encoder import EncoderBase
from ..decoder import DecoderBase
from ..toolkit import f_map_dim
from ..toolkit import auto_num_downsample
from ....types import losses_type
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import LossProtocol
from ....protocol import ModelProtocol
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Lambda


@LossProtocol.register("vae")
class VAELoss(LossProtocol):
    def _init_config(self) -> None:
        self.kld_ratio = self.config.setdefault("kld_ratio", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        **kwargs: Any,
    ) -> losses_type:
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu, log_var = map(forward_results.get, ["mu", "log_var"])
        assert mu is not None and log_var is not None
        var = log_var.exp()
        kld_losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - var, dim=1)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        loss = mse + self.kld_ratio * kld_loss
        return {"mse": mse, "kld": kld_loss, LOSS_KEY: loss}


class VanillaVAE(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_dim: int = 128,
        encoder_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
    ):
        super().__init__()
        self.img_size = img_size
        # encoder
        if encoder_configs is None:
            encoder_configs = {}
        encoder_configs["img_size"] = img_size
        encoder_configs["in_channels"] = in_channels
        self.encoder = EncoderBase.make(encoder, **encoder_configs)
        # latent
        latent_channels = self.encoder.latent_channels
        map_dim = f_map_dim(img_size, auto_num_downsample(img_size))
        out_flat_dim = latent_channels * map_dim ** 2
        self.to_latent = nn.Sequential(
            Lambda(lambda tensor: tensor.view(tensor.shape[0], -1), "flatten"),
            nn.Linear(out_flat_dim, 2 * latent_dim),
        )
        shape = -1, latent_channels, map_dim, map_dim
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, out_flat_dim),
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
        )
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = latent_channels
        decoder_configs["out_channels"] = out_channels or in_channels
        self.decoder = DecoderBase.make(decoder, **decoder_configs)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        encoding = self.encoder.encode(batch, **kwargs)[PREDICTIONS_KEY]
        net = self.to_latent(encoding)
        mu, log_var = net.chunk(2, dim=1)
        net = self.reparameterize(mu, log_var)
        batch[INPUT_KEY] = self.from_latent(net)
        decoded = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        net = F.interpolate(decoded, size=self.img_size)
        net = torch.tanh(net)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}


__all__ = [
    "VAELoss",
    "VanillaVAE",
]
