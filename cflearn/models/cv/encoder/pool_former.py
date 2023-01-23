from typing import Any
from typing import Dict
from typing import Optional

from .schema import Encoder1DFromPatches
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("pool_former")
class PoolFormerEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 4,
        norm_type: Optional[str] = "layer",
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            token_mixing_type="pool",
            num_layers=num_layers,
            norm_type=norm_type,
        )


__all__ = ["PoolFormerEncoder"]
