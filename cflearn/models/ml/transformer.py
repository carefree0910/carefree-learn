from typing import Any
from typing import Optional

from .protocol import MixedStackedModel
from .protocol import register_ml_module


@register_ml_module("transformer")
class Transformer(MixedStackedModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        feedforward_dim_ratio: float = 4.0,
        use_head_token: bool = False,
        **attention_kwargs: Any,
    ):
        attention_kwargs.setdefault("bias", False)
        attention_kwargs.setdefault("num_heads", 8)
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            use_head_token=use_head_token,
            use_positional_encoding=True,
        )


__all__ = ["Transformer"]
