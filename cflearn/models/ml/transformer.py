import torch.nn as nn

from typing import Any
from .stacks import MixedStackedModel
from .stacks import TokenMixerFactory
from ...modules.blocks import PreNorm
from ...modules.blocks import Attention


class AttentionTokenMixer(TokenMixerFactory):
    @staticmethod
    def make(
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        dropout: float,
        norm_type: str,
        **kwargs: Any,
    ) -> nn.Module:
        qkv_bias = kwargs.get("qkv_bias", False)
        num_heads = kwargs.get("num_heads", 8)
        return PreNorm(
            latent_dim,
            module=Attention.make(
                "basic",
                config=dict(
                    input_dim=latent_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    in_linear_config={"bias": qkv_bias},
                    is_self_attention=True,
                ),
            ),
            norm_type=norm_type,
        )


@MixedStackedModel.register("transformer")
class Transformer(MixedStackedModel):
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
        feedforward_dim_ratio: float = 4.0,
        use_head_token: bool = False,
        qkv_bias: bool = False,
        num_heads: int = 8,
    ):
        super().__init__(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            AttentionTokenMixer(),
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            use_head_token=use_head_token,
            use_positional_encoding=True,
            qkv_bias=qkv_bias,
            num_heads=num_heads,
        )


__all__ = ["Transformer"]
