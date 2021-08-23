from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....modules.blocks import AttentionTokenMixer
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("vit")
class ViTEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_configs: Optional[Dict[str, Any]] = None,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: str = "layer_norm",
        feedforward_dim_ratio: float = 4.0,
        bias: bool = False,
        num_heads: int = 8,
        **attention_kwargs: Any,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            to_patches_configs,
        )
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            AttentionTokenMixer(),
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            use_head_token=True,
            use_positional_encoding=True,
            bias=bias,
            num_heads=num_heads,
            **attention_kwargs,
        )


__all__ = ["ViTEncoder"]
