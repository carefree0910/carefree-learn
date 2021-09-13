import math
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
from ..decoder.vanilla import VanillaDecoder
from ....misc.toolkit import interpolate
from ....modules.blocks import Conv2d


@ModelProtocol.register("cascade_u2net")
class CascadeU2Net(CascadeBase):
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
        lv2_model_config["in_channels"] = in_channels + out_channels
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
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        lv1_outputs = self.lv1_net(batch_idx, batch, state, **kwargs)
        lv1_raw_alpha = lv1_outputs[PREDICTIONS_KEY][0]
        lv1_alpha = torch.sigmoid(lv1_raw_alpha)
        inp = batch[INPUT_KEY]
        resolution = lv1_alpha.shape[-1]
        if self.lv2_resolution is not None:
            lv2_res = self.lv2_resolution
            inp = interpolate(batch[INPUT_KEY], size=lv2_res, mode="bilinear")
            lv1_alpha = interpolate(lv1_alpha, size=lv2_res, mode="bilinear")
        lv2_input = torch.cat([inp, lv1_alpha], dim=1)
        lv2_outputs = self.lv2_net(batch_idx, {INPUT_KEY: lv2_input}, state, **kwargs)
        lv2_raw_alpha = lv2_outputs[PREDICTIONS_KEY]
        if isinstance(lv2_raw_alpha, list):
            lv2_raw_alpha = lv2_raw_alpha[0]
        if self.lv2_resolution is not None:
            lv2_raw_alpha = interpolate(lv2_raw_alpha, size=resolution, mode="bilinear")
        final_raw_alpha = lv1_raw_alpha + lv2_raw_alpha
        return {PREDICTIONS_KEY: final_raw_alpha}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        results = self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]
        if isinstance(results, list):
            results = results[0]
        return results


@ModelProtocol.register("cascade_dino")
class CascadeDINO(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        num_upsample: int,
        out_channels: int,
        *,
        dino_channels: int,
        latent_channels: int = 128,
        dino_name: str = "dino_vits8",
    ):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dino:main", dino_name)
        self.to_latent = Conv2d(
            dino_channels,
            latent_channels,
            kernel_size=1,
            bias=False,
        )
        self.decoder = VanillaDecoder(
            img_size,
            latent_channels,
            img_size // 2 ** num_upsample,
            num_upsample,
            out_channels,
        ).decoder

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        net = self.dino.get_last_selfattention(net)[..., 0, 1:]
        resolution = int(round(math.sqrt(net.shape[-1])))
        net = net.view(net.shape[0], -1, resolution, resolution)
        net = self.to_latent(net)
        return {PREDICTIONS_KEY: self.decoder(net)}

    def generate_from(self, net: Tensor, **kwargs: Any) -> Tensor:
        results = self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]
        if isinstance(results, list):
            results = results[0]
        return results


__all__ = [
    "CascadeU2Net",
    "CascadeDINO",
]