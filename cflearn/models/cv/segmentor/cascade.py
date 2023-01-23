import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.types import tensor_dict_type

from .constants import LV1_ALPHA_KEY
from .constants import LV1_RAW_ALPHA_KEY
from ...bases import CascadeMixin
from ....schema import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ...schemas.cv import ImageTranslatorMixin
from ....misc.toolkit import interpolate
from ....misc.internal_.register import register_module


@register_module("cascade_u2net")
class CascadeU2Net(nn.Module, CascadeMixin, ImageTranslatorMixin):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        lv2_name: str = "alpha_refine",
        lv2_resolution: Optional[int] = None,
        lv2_model_config: Optional[Dict[str, Any]] = None,
        lv1_model_trainable: Optional[bool] = None,
        lv1_model_ckpt_path: Optional[str] = None,
        latent_channels: int = 32,
        num_layers: int = 5,
        num_inner_layers: int = 7,
        lite: bool = False,
    ):
        super().__init__()
        if lv1_model_trainable is None:
            lv1_model_trainable = lv1_model_ckpt_path is None
        if not lv1_model_trainable and lv1_model_ckpt_path is None:
            raise ValueError("lv1 model should be trainable when ckpt is not provided")
        lv1_model_config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            num_layers=num_layers,
            num_inner_layers=num_inner_layers,
            lite=lite,
        )
        self.lv2_resolution = lv2_resolution
        if lv2_model_config is None:
            lv2_model_config = {}
        lv2_model_config["in_channels"] = in_channels
        lv2_model_config["out_channels"] = out_channels
        self._construct(
            "u2net",
            lv2_name,
            lv1_model_config,
            lv2_model_config,
            lv1_model_ckpt_path,
            lv1_model_trainable,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> Tensor:
        lv1_outputs = self.lv1_net(batch_idx, batch, state, **kwargs)
        lv1_raw_alpha = lv1_outputs[PREDICTIONS_KEY][0]
        lv1_alpha = torch.sigmoid(lv1_raw_alpha)
        inp = batch[INPUT_KEY]
        resolution = lv1_alpha.shape[-1]
        if self.lv2_resolution is not None:
            lv2_res = self.lv2_resolution
            inp = interpolate(batch[INPUT_KEY], size=lv2_res, mode="bilinear")
            lv1_raw_alpha = interpolate(lv1_raw_alpha, size=lv2_res, mode="bilinear")
            lv1_alpha = interpolate(lv1_alpha, size=lv2_res, mode="bilinear")
        lv2_outputs = self.lv2_net(
            batch_idx,
            {
                INPUT_KEY: inp,
                LV1_ALPHA_KEY: lv1_alpha,
                LV1_RAW_ALPHA_KEY: lv1_raw_alpha,
            },
            state,
            **kwargs,
        )
        lv2_raw_alpha = lv2_outputs[PREDICTIONS_KEY]
        if isinstance(lv2_raw_alpha, list):
            lv2_raw_alpha = lv2_raw_alpha[0]
        if self.lv2_resolution is not None:
            lv2_raw_alpha = interpolate(lv2_raw_alpha, size=resolution, mode="bilinear")
        return lv2_raw_alpha


__all__ = [
    "CascadeU2Net",
]
