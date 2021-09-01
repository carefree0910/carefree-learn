from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....modules.blocks import MLPTokenMixer
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("mixer")
class MixerEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_config: Optional[Dict[str, Any]] = None,
        *,
        num_layers: int = 4,
        norm_type: str = "layer_norm",
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            to_patches_config,
        )
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            MLPTokenMixer(),
            num_layers=num_layers,
            norm_type=norm_type,
        )


__all__ = ["MixerEncoder"]
