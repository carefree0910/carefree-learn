import torch.nn as nn

from typing import Any
from typing import Optional

from .stacks import FeedForward
from .stacks import MixedStackedModel
from .stacks import TokenMixerFactory
from .protocol import MERGED_KEY
from .protocol import MLCoreProtocol
from ..bases import RDropoutBase
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...modules.blocks import Lambda


class MLPTokenMixer(TokenMixerFactory):
    @staticmethod
    def make(
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> nn.Module:
        return nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2), name="to_token_mixing"),
            FeedForward(num_tokens, num_tokens, dropout),
            Lambda(lambda x: x.transpose(1, 2), name="to_channel_mixing"),
        )


@MixedStackedModel.register("mixer")
class Mixer(MixedStackedModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: str = "batch_norm",
    ):
        super().__init__(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            MLPTokenMixer(),
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


@MLCoreProtocol.register("mixer_r_dropout")
class MixerWithRDropout(RDropoutBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: str = "batch_norm",
        lb: float = 0.1,
        is_classification: bool,
    ):
        super().__init__()
        self.mixer = Mixer(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
        )
        # R-Dropout
        self.lb = lb
        self.is_classification = is_classification

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch[MERGED_KEY] = batch[INPUT_KEY]
        return self.mixer(batch_idx, batch, state, **kwargs)


__all__ = ["Mixer"]
