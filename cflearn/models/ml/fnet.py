import torch.nn as nn

from typing import Any
from torch.fft import fft

from .stacks import MixedStackedModel
from .stacks import TokenMixerFactory
from ...modules.blocks import Lambda


class FourierTokenMixer(TokenMixerFactory):
    @staticmethod
    def make(
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> nn.Module:
        return Lambda(lambda x: fft(fft(x, dim=-1), dim=-2).real, name="fourier")


@MixedStackedModel.register("fnet")
class FNet(MixedStackedModel):
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
            FourierTokenMixer(),
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


__all__ = ["FNet"]
