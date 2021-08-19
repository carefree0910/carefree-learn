import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..encoder import Encoder1DBase
from ..decoder import DecoderBase
from ..toolkit import f_map_dim
from ..toolkit import auto_num_layers
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import INFO_PREFIX
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import interpolate
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Linear
from ....modules.blocks import ChannelPadding


@ModelProtocol.register("vae")
class VanillaVAE(ModelProtocol, GaussianGeneratorMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        latent_channels: int = 16,
        target_downsample: int = 4,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder1d: str = "vanilla",
        decoder: str = "vanilla",
        encoder1d_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # dimensions
        self.num_classes = num_classes
        if num_downsample is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `num_downsample` should be provided "
                    "(when `num_downsample` is not provided, it will be inferred "
                    "automatically with `img_size` & `min_size`)"
                )
            num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        if latent_resolution is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `map_dim` should be provided "
                    "(when `map_dim` is not provided, it will be inferred "
                    "automatically with `img_size`)"
                )
            latent_resolution = f_map_dim(img_size, num_downsample)
        if img_size is None:
            raw_size = latent_resolution * 2 ** num_downsample
            print(f"{INFO_PREFIX}img_size is not provided, raw_size will be {raw_size}")
        self.latent_dim = latent_channels * latent_resolution ** 2
        # encoder
        if encoder1d_configs is None:
            encoder1d_configs = {}
        encoder1d_configs["img_size"] = img_size
        encoder1d_configs["in_channels"] = in_channels
        encoder1d_configs["latent_dim"] = self.latent_dim
        if encoder1d == "vanilla":
            encoder1d_configs["num_downsample"] = num_downsample
        self.encoder = Encoder1DBase.make(encoder1d, config=encoder1d_configs)
        self.to_statistics = Linear(self.latent_dim, 2 * self.latent_dim, bias=False)
        # latent
        shape = -1, latent_channels, latent_resolution, latent_resolution
        blocks: List[nn.Module] = [
            Lambda(lambda tensor: tensor.view(*shape), f"reshape -> {shape}"),
            Conv2d(latent_channels, self.latent_dim, kernel_size=1, bias=False),
        ]
        latent_dim = self.latent_dim
        if latent_padding_channels is not None:
            latent_dim += latent_padding_channels
            latent_padding = ChannelPadding(latent_padding_channels, latent_resolution)
            blocks.append(latent_padding)
        self.from_latent = nn.Sequential(*blocks)
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = latent_dim
        decoder_configs["latent_resolution"] = latent_resolution
        decoder_configs["num_upsample"] = num_downsample
        decoder_configs["out_channels"] = out_channels or in_channels
        decoder_configs["num_classes"] = num_classes
        self.decoder = DecoderBase.make(decoder, config=decoder_configs)

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @property
    def can_reconstruct(self) -> bool:
        return True

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        batch = {INPUT_KEY: self.from_latent(z), LABEL_KEY: labels}
        net = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        inp = batch[INPUT_KEY]
        net = self.encoder.encode(batch, **kwargs)[LATENT_KEY]
        net = self.to_statistics(net)
        mu, log_var = net.chunk(2, dim=1)
        net = self.reparameterize(mu, log_var)
        labels = None if self.num_classes is None else batch[LABEL_KEY].view(-1)
        net = self.decode(net, labels=labels, **kwargs)
        net = interpolate(net, anchor=inp)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}


__all__ = ["VanillaVAE"]
