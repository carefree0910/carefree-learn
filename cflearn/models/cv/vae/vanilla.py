import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional

from ..encoder import Encoder1DBase
from ..decoder import DecoderBase
from ..toolkit import f_map_dim
from ..toolkit import auto_num_layers
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Linear


@ModelProtocol.register("vae")
class VanillaVAE(ModelProtocol):
    condition_tokens: Optional[nn.Parameter]

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        min_size: int = 2,
        target_downsample: int = 4,
        latent_dim: int = 256,
        num_classes: Optional[int] = None,
        encoder1d_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
        *,
        encoder1d: str = "vanilla",
        decoder: str = "vanilla",
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        map_dim = f_map_dim(img_size, num_downsample)
        map_area = map_dim ** 2
        if latent_dim % map_area != 0:
            msg = f"`latent_dim` should be divided by `map_area` ({map_area})"
            raise ValueError(msg)
        # encoder
        if encoder1d_configs is None:
            encoder1d_configs = {}
        encoder1d_configs["img_size"] = img_size
        encoder1d_configs["in_channels"] = in_channels
        encoder1d_configs["latent_dim"] = latent_dim
        if encoder1d == "vanilla":
            encoder1d_configs["num_downsample"] = num_downsample
        self.encoder = Encoder1DBase.make(encoder1d, **encoder1d_configs)
        self.to_statistics = Linear(latent_dim, 2 * latent_dim, bias=False)
        # latent
        compressed_channels = latent_dim // map_area
        shape = -1, compressed_channels, map_dim, map_dim
        blocks = [Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}")]
        if num_classes is None:
            self.condition_tokens = None
        else:
            compressed_channels += num_classes
            token_shape = num_classes, map_dim, map_dim
            self.condition_tokens = nn.Parameter(torch.randn(1, *token_shape))
            blocks.append(
                Lambda(
                    lambda tensor: torch.cat(
                        [
                            tensor,
                            self.condition_tokens.repeat(tensor.shape[0], 1, 1, 1),  # type: ignore
                        ],
                        dim=1,
                    ),
                    f"concat ({num_classes}) condition tokens",
                )
            )
        self.from_latent = nn.Sequential(
            *blocks,
            Conv2d(compressed_channels, latent_dim, kernel_size=1, bias=False),
        )
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = latent_dim
        decoder_configs["latent_resolution"] = map_dim
        decoder_configs["num_upsample"] = num_downsample
        decoder_configs["out_channels"] = out_channels or in_channels
        self.decoder = DecoderBase.make(decoder, **decoder_configs)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch = {INPUT_KEY: self.from_latent(z)}
        net = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.encoder.encode(batch, **kwargs)[PREDICTIONS_KEY]
        net = self.to_statistics(net)
        mu, log_var = net.chunk(2, dim=1)
        net = self.reparameterize(mu, log_var)
        net = self._decode(net, **kwargs)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}

    def reconstruct(self, net: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]

    def sample(self, num_sample: int, **kwargs: Any) -> torch.Tensor:
        z = torch.randn(num_sample, self.latent_dim).to(self.device)
        return self._decode(z, **kwargs)


__all__ = ["VanillaVAE"]
