from torch import nn
from torch import Tensor
from typing import Any
from typing import Optional

from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import PerceiverIO
from ....modules.blocks import ImgToPatches


@ModelProtocol.register("perceiver_io_generator")
class PerceiverIOGenerator(ModelProtocol):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 128,
        *,
        num_layers: int = 6,
        num_latents: int = 64,
        num_cross_heads: int = 1,
        num_latent_heads: int = 8,
        cross_latent_dim: Optional[int] = None,
        self_latent_dim: Optional[int] = None,
        feedforward_dropout: float = 0.0,
        feedforward_dim_ratio: float = 4.0,
        reuse_weights: bool = False,
        num_self_attn_repeat: int = 1,
    ):
        super().__init__()
        self.to_patches = ImgToPatches(patch_size, patch_size, in_channels, latent_dim)
        self.to_latent = nn.Linear(in_channels, latent_dim)
        self.from_latent = nn.Linear(latent_dim, out_channels)
        self.net = PerceiverIO(
            input_dim=latent_dim,
            num_layers=num_layers,
            num_latents=num_latents,
            output_dim=latent_dim,
            latent_dim=latent_dim,
            num_cross_heads=num_cross_heads,
            num_latent_heads=num_latent_heads,
            cross_latent_dim=cross_latent_dim,
            self_latent_dim=self_latent_dim,
            feedforward_dropout=feedforward_dropout,
            feedforward_dim_ratio=feedforward_dim_ratio,
            reuse_weights=reuse_weights,
            num_self_attn_repeat=num_self_attn_repeat,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        inp = batch[INPUT_KEY]
        b, c, h, w = inp.shape
        net = self.to_patches(inp)
        inp = inp.view(b, c, h * w).transpose(1, 2).contiguous()
        inp = self.to_latent(inp)
        net = self.net(net, out_queries=inp)
        net = self.from_latent(net)
        net = net.transpose(1, 2).contiguous().view(b, -1, h, w)
        return {PREDICTIONS_KEY: net}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]


__all__ = ["PerceiverIO"]
