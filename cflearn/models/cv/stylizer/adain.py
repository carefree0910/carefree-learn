from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import print_warning
from cftool.types import tensor_dict_type

from ..decoder import VanillaDecoder
from ..encoder import BackboneEncoder
from ....types import losses_type
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import STYLE_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ....constants import STYLE_LATENTS_KEY
from ....constants import CONTENT_LATENT_KEY
from ....constants import STYLIZED_STYLE_LATENTS_KEY
from ....constants import STYLIZED_CONTENT_LATENT_KEY
from ....misc.toolkit import adain_with_tensor
from ....misc.toolkit import mean_std
from ....misc.toolkit import interpolate
from ....misc.internal_.register import register_module
from ....misc.internal_.register import register_loss_module

try:
    from cfcv.misc.toolkit import clip_normalize
except:
    clip_normalize = None


@register_module("adain")
class AdaINStylizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        *,
        backbone: str = "vgg_style",
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
            num_repeats="repeated",
            num_upsample=self.backbone.num_downsample,
            **(decoder_config or {}),
        )

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        style = batch[STYLE_KEY]
        content = batch[INPUT_KEY]
        determinate = kwargs.pop("determinate", False)
        need_stylized_features = kwargs.pop("need_stylized_features", True)
        style_feats = self.backbone(style)
        style_latent = style_feats.pop(LATENT_KEY)
        content_feats = self.backbone(content)
        content_latent = content_feats[LATENT_KEY]
        args = content_latent, style_latent
        encoded = adain_with_tensor(*args, determinate=determinate)
        style_weight = kwargs.get("style_weight", 1.0)
        encoded = style_weight * encoded + (1.0 - style_weight) * content_latent
        net = self.decoder({INPUT_KEY: encoded}, determinate=determinate, **kwargs)
        decoded = interpolate(net, anchor=content)
        if not need_stylized_features:
            decoded_feats = decoded_content_latent = None
        else:
            decoded_feats = self.backbone(decoded)
            decoded_content_latent = decoded_feats.pop(LATENT_KEY)
        return {
            PREDICTIONS_KEY: decoded,
            STYLE_LATENTS_KEY: style_feats,
            CONTENT_LATENT_KEY: encoded,
            STYLIZED_STYLE_LATENTS_KEY: decoded_feats,
            STYLIZED_CONTENT_LATENT_KEY: decoded_content_latent,
        }

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.stylize(batch[INPUT_KEY], batch[STYLE_KEY], determinate=True)

    def stylize(self, net: Tensor, style: Tensor, **kwargs: Any) -> Tensor:
        inp = {INPUT_KEY: net, STYLE_KEY: style}
        kwargs["need_stylized_features"] = False
        decoded = self.forward(inp, **kwargs)[PREDICTIONS_KEY]
        if clip_normalize is not None:
            return clip_normalize(decoded)
        print_warning(
            "`carefree-cv` is not installed, "
            "so the stylized image is not normalized, which might harm the quality"
        )
        return decoded


@register_loss_module("adain")
class AdaINLoss(nn.Module):
    def __init__(self, *, content_weight: float = 1.0, style_weight: float = 10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.content_w = content_weight
        self.style_w = style_weight

    def _content_loss(self, stylized: Tensor, target: Tensor) -> Tensor:
        return self.mse(stylized, target)

    def _style_loss(self, stylized: Tensor, target: Tensor) -> Tensor:
        stylized_mean, stylized_std = mean_std(stylized)
        target_mean, target_std = mean_std(target)
        mean_loss = self.mse(stylized_mean, target_mean)
        std_loss = self.mse(stylized_std, target_std)
        return mean_loss + std_loss

    def forward(self, forward_results: tensor_dict_type) -> losses_type:
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
