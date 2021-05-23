from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ...ml.mixer import MLPTokenMixer
from ...ml.stacks import MixedStackedEncoder


@Encoder1DFromPatches.register("mixer")
class MixerEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        to_patches_configs: Optional[Dict[str, Any]] = None,
        *,
        num_layers: int = 4,
        norm_type: str = "layer_norm",
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
            MLPTokenMixer(),
            num_layers=num_layers,
            norm_type=norm_type,
        )

    def from_patches(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {LATENT_KEY: self.encoder(batch[INPUT_KEY])}


__all__ = ["MixerEncoder"]
