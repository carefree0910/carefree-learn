from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from cftool.array import squeeze
from cftool.types import tensor_dict_type

from .core import Backbone
from ..schema import EncoderMixin
from ..schema import Encoder1DMixin
from .....constants import LATENT_KEY
from .....modules.blocks import Conv2d


class Preset:
    remove_layers: Dict[str, List[str]] = {}
    target_layers: Dict[str, Dict[str, str]] = {}
    increment_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_settings(cls) -> Callable:
        def _register(settings: Preset) -> None:
            cls.remove_layers.update(settings.remove_layers)
            cls.target_layers.update(settings.target_layers)
            cls.increment_configs.update(settings.increment_configs)

        return _register


@EncoderMixin.register("backbone")
class BackboneEncoder(nn.Module, EncoderMixin):
    def __init__(
        self,
        name: str,
        in_channels: int,
        *,
        finetune: bool = True,
        pretrained: bool = False,
        use_to_rgb: bool = False,
        to_rgb_bias: bool = False,
        backbone_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        if in_channels == 3 and not use_to_rgb:
            self.to_rgb = None
        else:
            self.to_rgb = Conv2d(in_channels, 3, kernel_size=1, bias=to_rgb_bias)
        self.net = Backbone(
            name,
            pretrained=pretrained,
            requires_grad=finetune,
            **(backbone_config or {}),
        )
        self.num_downsample = self.net.num_downsample
        self.latent_channels = self.net.latent_channels

    def forward(self, net: Tensor) -> tensor_dict_type:
        if self.to_rgb is not None:
            net = self.to_rgb(net)
        return self.net(net)


@Encoder1DMixin.register("backbone")
class BackboneEncoder1D(nn.Module, Encoder1DMixin):
    def __init__(
        self,
        name: str,
        in_channels: int,
        *,
        finetune: bool = True,
        pretrained: bool = False,
        backbone_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = BackboneEncoder(
            name,
            in_channels,
            finetune=finetune,
            pretrained=pretrained,
            backbone_config=backbone_config,
        )
        self.latent_dim = self.encoder.latent_channels
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, net: Tensor) -> tensor_dict_type:
        outputs = self.encoder(net)
        latent = outputs[LATENT_KEY]
        if latent.shape[-2] != 1 or latent.shape[-1] != 1:
            latent = self.pool(latent)
        outputs[LATENT_KEY] = squeeze(latent)
        return outputs


__all__ = [
    "BackboneEncoder",
    "BackboneEncoder1D",
]
