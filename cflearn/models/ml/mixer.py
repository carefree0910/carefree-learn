from typing import Any
from typing import Dict
from typing import Optional

from .protocol import MERGED_KEY
from .protocol import MixedStackedModel
from ..bases import BAKEBase
from ..bases import RDropoutBase
from ...types import tensor_dict_type
from ...protocol import TrainerState
from ...constants import INPUT_KEY
from ...constants import LATENT_KEY
from ...constants import PREDICTIONS_KEY


@MixedStackedModel.register("mixer")
class Mixer(MixedStackedModel):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
    ):
        super().__init__(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            token_mixing_type="mlp",
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


@MixedStackedModel.register("mixer_bake")
class MixerWithBAKE(BAKEBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.1,
        norm_type: Optional[str] = "batch_norm",
        lb: float = 0.1,
        bake_loss: str = "auto",
        bake_loss_config: Optional[Dict[str, Any]] = None,
        w_ensemble: float = 0.5,
        is_classification: bool,
    ):
        super().__init__()
        self.mixer = Mixer(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
        )
        self._init_bake(lb, bake_loss, bake_loss_config, w_ensemble, is_classification)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        net = self.mixer.to_encoder(net)
        latent = self.mixer.encoder(net)
        net = self.mixer.head(latent)
        return {LATENT_KEY: latent, PREDICTIONS_KEY: net}


@MixedStackedModel.register("mixer_r_dropout")
class MixerWithRDropout(RDropoutBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int = 256,
        *,
        num_layers: int = 4,
        dropout: float = 0.1,
        norm_type: Optional[str] = "batch_norm",
        lb: float = 0.1,
        is_classification: bool,
    ):
        super().__init__()
        self.mixer = Mixer(
            in_dim,
            out_dim,
            num_history,
            latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
        )
        # R-Dropout
        self.lb = lb
        self.is_classification = is_classification

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch[MERGED_KEY] = batch[INPUT_KEY]
        return self.mixer(batch_idx, batch, state, **kwargs)


__all__ = [
    "Mixer",
    "MixerWithBAKE",
    "MixerWithRDropout",
]
