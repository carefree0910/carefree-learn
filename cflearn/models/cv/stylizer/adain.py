from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .constants import STYLE_KEY
from .constants import STYLE_LATENTS_KEY
from .constants import CONTENT_LATENT_KEY
from .constants import STYLIZED_STYLE_LATENTS_KEY
from .constants import STYLIZED_CONTENT_LATENT_KEY
from ..decoder import VanillaDecoder
from ..encoder import BackboneEncoder
from ....types import losses_type
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import LossProtocol
from ....protocol import ModelProtocol
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import interpolate
from ....misc.toolkit import quantile_normalize


def mean_std(latent_map: Tensor, eps: float = 1.0e-5) -> Tuple[Tensor, Tensor]:
    n, c = latent_map.shape[:2]
    latent_var = latent_map.view(n, c, -1).var(dim=2) + eps
    latent_std = latent_var.sqrt().view(n, c, 1, 1)
    latent_mean = latent_map.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return latent_mean, latent_std


def adain(content_latent: Tensor, style_latent: Tensor) -> Tensor:
    style_mean, style_std = mean_std(style_latent)
    content_mean, content_std = mean_std(content_latent)
    content_normalized = (content_latent - content_mean) / content_std
    return content_normalized * style_std + style_mean


@ModelProtocol.register("adain")
class AdaINStylizer(ModelProtocol):
    def __init__(
        self,
        in_channels: int = 3,
        *,
        backbone: str = "vgg19_lite",
        backbone_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = BackboneEncoder(
            backbone,
            in_channels,
            finetune=False,
            pretrained=True,
            backbone_config=backbone_config,
        )
        if decoder_config is None:
            decoder_config = {}
        decoder_config.setdefault("norm_type", None)
        decoder_config.setdefault("activation", "relu")
        self.decoder = VanillaDecoder(
            self.backbone.latent_channels,
            in_channels,
            num_upsample=self.backbone.num_downsample,
            **(decoder_config or {}),
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        style = batch[STYLE_KEY]
        content = batch[INPUT_KEY]
        style_feats = self.backbone(batch_idx, {INPUT_KEY: style}, state, **kwargs)
        style_latent = style_feats.pop(LATENT_KEY)
        content_feats = self.backbone(batch_idx, {INPUT_KEY: content}, state, **kwargs)
        content_latent = content_feats[LATENT_KEY]
        encoded = adain(content_latent, style_latent)
        style_weight = kwargs.get("style_weight", 1.0)
        encoded = style_weight * encoded + (1.0 - style_weight) * content_latent
        rs = self.decoder(batch_idx, {INPUT_KEY: encoded}, state, **kwargs)
        decoded = interpolate(rs[PREDICTIONS_KEY], anchor=content)
        decoded_feats = self.backbone(batch_idx, {INPUT_KEY: decoded}, state, **kwargs)
        decoded_content_latent = decoded_feats.pop(LATENT_KEY)
        return {
            PREDICTIONS_KEY: decoded,
            STYLE_LATENTS_KEY: style_feats,
            CONTENT_LATENT_KEY: encoded,
            STYLIZED_STYLE_LATENTS_KEY: decoded_feats,
            STYLIZED_CONTENT_LATENT_KEY: decoded_content_latent,
        }

    def stylize(self, net: Tensor, style: Tensor, **kwargs: Any) -> Tensor:
        inp = {INPUT_KEY: net, STYLE_KEY: style}
        decoded = self.forward(0, inp, **kwargs)[PREDICTIONS_KEY]
        return quantile_normalize(decoded, **kwargs)


@LossProtocol.register("adain")
class AdaINLoss(LossProtocol):
    def _init_config(self) -> None:
        self.mse = nn.MSELoss()
        self.content_w = self.config.setdefault("content_weight", 1.0)
        self.style_w = self.config.setdefault("style_weight", 10.0)

    def _content_loss(self, stylized: Tensor, target: Tensor) -> Tensor:
        return self.mse(stylized, target)

    def _style_loss(self, stylized: Tensor, target: Tensor) -> Tensor:
        stylized_mean, stylized_std = mean_std(stylized)
        target_mean, target_std = mean_std(target)
        mean_loss = self.mse(stylized_mean, target_mean)
        std_loss = self.mse(stylized_std, target_std)
        return mean_loss + std_loss

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        content_latent = forward_results[CONTENT_LATENT_KEY]
        stylized_latent = forward_results[STYLIZED_CONTENT_LATENT_KEY]
        content_loss = self._content_loss(stylized_latent, content_latent)
        style_feats = forward_results[STYLE_LATENTS_KEY]
        stylized_feats = forward_results[STYLIZED_STYLE_LATENTS_KEY]
        style_loss = sum(
            self._style_loss(stylized, style_feats[k])
            for k, stylized in stylized_feats.items()
        )
        loss = self.content_w * content_loss + self.style_w * style_loss
        return {LOSS_KEY: loss, "content": content_loss, "style": style_loss}


__all__ = [
    "AdaINStylizer",
    "AdaINLoss",
]
