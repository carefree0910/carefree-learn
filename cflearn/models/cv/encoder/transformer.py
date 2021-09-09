from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("vit")
class ViTEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 384,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        *,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        sequence_pool: bool = False,
        use_head_token: bool = True,
        use_positional_encoding: bool = True,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            to_patches_type,
            to_patches_config,
        )
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", 6)
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            residual_after_norm=residual_after_norm,
            feedforward_dim_ratio=feedforward_dim_ratio,
            sequence_pool=sequence_pool,
            use_head_token=use_head_token,
            use_positional_encoding=use_positional_encoding,
        )


__all__ = ["ViTEncoder"]
