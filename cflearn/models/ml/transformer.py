from typing import Any

from .protocol import MixedStackedModel
from ...modules.blocks import AttentionTokenMixer


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
        bias: bool = False,
        num_heads: int = 8,
        **attention_kwargs: Any,
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
            bias=bias,
            num_heads=num_heads,
            **attention_kwargs,
        )


__all__ = ["Transformer"]
