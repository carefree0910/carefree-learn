import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ...ml.mixer import MixerBlock
from ....modules.blocks import _get_clones
from ....modules.blocks import Lambda
from ....modules.blocks import PreNorm


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
        mixer_block = MixerBlock(self.to_patches.num_patches, latent_dim, norm_type)
        self.encoder = nn.Sequential(
            *_get_clones(mixer_block, num_layers, return_list=True),
            PreNorm(
                latent_dim,
                module=Lambda(lambda x: x.mean(1), name="global_average"),
                norm_type=norm_type,
            ),
        )

    def from_patches(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.encoder(batch[INPUT_KEY])}


__all__ = ["MixerEncoder"]
