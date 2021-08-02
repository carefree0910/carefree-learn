import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ...bases import CascadeBase
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import align_to
from ....misc.toolkit import imagenet_normalize


@ModelProtocol.register("cascade_u2net")
class CascadeU2Net(CascadeBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        lv1_model_ckpt_path: str,
        lv2_resolution: Optional[int] = None,
        lv2_model_config: Optional[Dict[str, Any]] = None,
        latent_channels: int = 32,
        num_layers: int = 5,
        max_layers: int = 7,
        lite: bool = False,
    ):
        super().__init__()
        lv1_model_config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            max_layers=max_layers,
            lite=lite,
        )
        self.lv2_resolution = lv2_resolution
        if lv2_model_config is None:
            lv2_model_config = {}
        for k, v in lv1_model_config.items():
            lv2_model_config.setdefault(k, v)
        lv2_model_config["in_channels"] = in_channels + out_channels
        self._construct(
            "u2net",
            "u2net",
            lv1_model_config,
            lv2_model_config,
            lv1_model_ckpt_path,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        lv1_outputs = self.lv1_net(batch_idx, batch, state, **kwargs)
        lv1_alpha = torch.sigmoid(lv1_outputs[PREDICTIONS_KEY][0])
        lv1_alpha = imagenet_normalize(lv1_alpha)
        inp = batch[INPUT_KEY]
        resolution = lv1_alpha.shape[-1]
        if self.lv2_resolution is not None:
            inp = align_to(batch[INPUT_KEY], size=self.lv2_resolution)
            lv1_alpha = align_to(lv1_alpha, size=self.lv2_resolution)
        lv2_input = torch.cat([inp, lv1_alpha], dim=1)
        lv2_outputs = self.lv2_net(batch_idx, {INPUT_KEY: lv2_input}, state, **kwargs)
        if self.lv2_resolution is None:
            return lv2_outputs
        lv2_logits = lv2_outputs[PREDICTIONS_KEY]
        lv2_outputs = [align_to(logits, size=resolution) for logits in lv2_logits]
        return {PREDICTIONS_KEY: lv2_outputs}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY][0]


__all__ = [
    "CascadeU2Net",
]
