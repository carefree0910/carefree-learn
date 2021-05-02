import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import Optional

from ..encoder import EncoderBase
from ..decoder import DecoderBase
from ..toolkit import f_map_dim
from ..toolkit import auto_num_downsample
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda


class VanillaVAE(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        min_size: int = 4,
        target_downsample: int = 4,
        latent_dim: int = 128,
        encoder_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
    ):
        super().__init__()
        self.img_size = img_size
        num_downsample = auto_num_downsample(img_size, min_size, target_downsample)
        # encoder
        if encoder_configs is None:
            encoder_configs = {}
        encoder_configs["img_size"] = img_size
        encoder_configs["in_channels"] = in_channels
        encoder_configs["num_downsample"] = num_downsample
        self.encoder = EncoderBase.make(encoder, **encoder_configs)
        # latent
        self.latent_dim = latent_dim
        latent_channels = self.encoder.latent_channels
        map_dim = f_map_dim(img_size, num_downsample)
        map_area = map_dim ** 2
        if (latent_dim * 2) % map_area != 0:
            msg = f"`latent_dim` should be divided by `map_area` ({map_area})"
            raise ValueError(msg)
        compressed_channels = latent_dim // map_area
        self.to_latent = nn.Sequential(
            Conv2d(latent_channels, 2 * compressed_channels, kernel_size=1),
            Lambda(lambda tensor: tensor.view(-1, 2 * latent_dim), "flatten"),
        )
        shape = -1, compressed_channels, map_dim, map_dim
        self.from_latent = nn.Sequential(
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(compressed_channels, latent_channels, kernel_size=1),
        )
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = latent_channels
        decoder_configs["num_upsample"] = num_downsample
        decoder_configs["out_channels"] = out_channels or in_channels
        self.decoder = DecoderBase.make(decoder, **decoder_configs)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        batch = {INPUT_KEY: self.from_latent(z)}
        decoded = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        net = F.interpolate(decoded, size=self.img_size)
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        encoding = self.encoder.encode(batch, **kwargs)[PREDICTIONS_KEY]
        net = self.to_latent(encoding)
        mu, log_var = net.chunk(2, dim=1)
        net = self.reparameterize(mu, log_var)
        net = self._decode(net, **kwargs)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}

    def reconstruct(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(0, {INPUT_KEY: tensor}, **kwargs)[PREDICTIONS_KEY]

    def sample(self, num_sample: int, **kwargs: Any) -> torch.Tensor:
        z = torch.randn(num_sample, self.latent_dim)
        return self._decode(z, **kwargs)


__all__ = ["VanillaVAE"]
