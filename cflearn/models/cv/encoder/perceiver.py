from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from .schema import Encoder1DFromPatches
from ....modules.blocks import PerceiverIO


class PerceiverIOForEncoder1D(PerceiverIO):
    def forward(
        self,
        net: Tensor,
        *,
        mask: Optional[Tensor] = None,
        out_queries: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        net = super().forward(net, mask=mask, out_queries=out_queries)
        return net.squeeze(1)


@Encoder1DFromPatches.register("perceiver_io")
class PerceiverIOEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 6,
        num_latents: int = 64,
        output_dim: Optional[int] = None,
        num_cross_heads: int = 1,
        num_latent_heads: int = 8,
        cross_latent_dim: Optional[int] = None,
        self_latent_dim: Optional[int] = None,
        feedforward_dropout: float = 0.0,
        feedforward_dim_ratio: float = 4.0,
        reuse_weights: bool = False,
        num_self_attn_repeat: int = 1,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        self.encoder = PerceiverIOForEncoder1D(
            input_dim=latent_dim,
            num_layers=num_layers,
            num_latents=num_latents,
            output_dim=output_dim,
            num_output=1,
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


__all__ = [
    "PerceiverIOEncoder",
]
