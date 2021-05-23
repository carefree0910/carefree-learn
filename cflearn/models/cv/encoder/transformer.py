from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ...ml.transformer import TransformerEncoder
from ....constants import LATENT_KEY


@Encoder1DFromPatches.register("vit")
class ViTEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        feed_forward_dim: int = 512,
        to_patches_configs: Optional[Dict[str, Any]] = None,
        *,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        norm_type: str = "batch_norm",
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            to_patches_configs,
        )
        self.encoder = TransformerEncoder(
            latent_dim,
            num_heads,
            self.to_patches.num_patches,
            feed_forward_dim,
            num_layers=num_layers,
            dropout=dropout,
            qkv_bias=qkv_bias,
            norm_type=norm_type,
            use_head_token=True,
        )

    def from_patches(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {LATENT_KEY: self.encoder(batch[INPUT_KEY])}


__all__ = ["ViTEncoder"]
