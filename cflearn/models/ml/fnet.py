from typing import Optional

from ..register import register_ml_module
from ..protocols.ml import MixedStackedModel


@register_ml_module("fnet")
class FNet(MixedStackedModel):
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
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim,
            token_mixing_type="fourier",
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


__all__ = ["FNet"]
