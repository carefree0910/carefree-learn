import torch

import torch.nn as nn

from typing import Any
from typing import Optional
from cflearn.types import tensor_dict_type
from cflearn.trainer import TrainerState
from cflearn.constants import PREDICTIONS_KEY
from cflearn.modules.blocks import _get_clones
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import Linear
from cflearn.modules.blocks import PreNorm
from cflearn.modules.blocks import Residual
from cflearn.models.ml.protocol import MERGED_KEY
from cflearn.models.ml.protocol import MLCoreProtocol


class MLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(Linear(dim, dim), nn.GELU(), Linear(dim, dim))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.net(net)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, latent_dim: int, norm_type: str = "batch_norm"):
        super().__init__()
        self.token_mixing = Residual(
            PreNorm(
                latent_dim,
                module=nn.Sequential(
                    Lambda(lambda x: x.transpose(1, 2), name="to_token_mixing"),
                    MLPBlock(num_tokens),
                    Lambda(lambda x: x.transpose(1, 2), name="to_channel_mixing"),
                ),
                norm_type=norm_type,
            )
        )
        self.channel_mixing = Residual(
            PreNorm(
                latent_dim,
                module=MLPBlock(latent_dim),
                norm_type=norm_type,
            )
        )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.channel_mixing(self.token_mixing(net))


@MLCoreProtocol.register("mixer")
class Mixer(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        *,
        num_layers: int = 4,
        latent_dim: int = 256,
        norm_type: str = "batch_norm",
    ):
        super().__init__(in_dim, out_dim, num_history)
        self.to_mixer = Linear(in_dim, latent_dim)
        mixer_block = MixerBlock(num_history, latent_dim, norm_type)
        self.mixer_blocks = _get_clones(mixer_block, num_layers)
        self.pre_head = PreNorm(
            latent_dim,
            module=Lambda(lambda x: x.mean(1), name="global_average"),
            norm_type="layer_norm",
        )
        self.head = Linear(latent_dim, out_dim)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[MERGED_KEY]
        net = self.to_mixer(net)
        for block in self.mixer_blocks:
            net = block(net)
        net = self.pre_head(net)
        net = self.head(net)
        return {PREDICTIONS_KEY: net}


__all__ = ["Mixer"]
