from torch import nn
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional

from .core import Backbone
from ..protocol import EncoderBase
from ..protocol import Encoder1DBase
from .....types import tensor_dict_type
from .....trainer import TrainerState
from .....constants import INPUT_KEY
from .....constants import LATENT_KEY
from .....constants import WARNING_PREFIX
from .....misc.toolkit import squeeze
from .....modules.blocks import Conv2d


class Preset:
    remove_layers: Dict[str, List[str]] = {}
    target_layers: Dict[str, Dict[str, str]] = {}
    latent_dims: Dict[str, int] = {}
    num_downsample: Dict[str, int] = {}
    increment_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_settings(cls) -> Callable:
        def _register(settings: Preset) -> None:
            cls.remove_layers.update(settings.remove_layers)
            cls.target_layers.update(settings.target_layers)
            cls.latent_dims.update(settings.latent_dims)
            cls.num_downsample.update(settings.num_downsample)
            cls.increment_configs.update(settings.increment_configs)

        return _register


@EncoderBase.register("backbone")
class BackboneEncoder(EncoderBase):
    def __init__(
        self,
        name: str,
        in_channels: int,
        img_size: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_downsample: Optional[int] = None,
        *,
        finetune: bool = True,
        pretrained: bool = False,
        need_normalize: bool = False,
        remove_layers: Optional[List[str]] = None,
        target_layers: Optional[Dict[str, str]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ):
        # preset stuffs
        if img_size is not None:
            print(f"{WARNING_PREFIX}`img_size` will not affect `BackboneEncoder`")
        if remove_layers is None:
            remove_layers = Preset.remove_layers.get(name)
            if remove_layers is None:
                msg = f"`remove_layers` should be provided for `{name}`"
                raise ValueError(msg)
        if target_layers is None:
            target_layers = Preset.target_layers.get(name)
            if target_layers is None:
                msg = f"`target_layers` should be provided for `{name}`"
                raise ValueError(msg)
        preset_dim = Preset.latent_dims.get(name)
        if latent_dim is None:
            latent_dim = preset_dim
            if latent_dim is None:
                msg = f"`latent_dim` should be provided for `{name}`"
                raise ValueError(msg)
        else:
            if preset_dim is not None and latent_dim != preset_dim:
                raise ValueError(
                    f"provided `latent_dim` ({latent_dim}) is not "
                    f"identical with `preset_dim` ({preset_dim}), "
                    f"please consider set `latent_dim` to {preset_dim}"
                )
        self.latent_dim = latent_dim
        if num_downsample is not None:
            preset_downsample = Preset.num_downsample.get(name)
            if preset_downsample is not None and num_downsample != preset_downsample:
                raise ValueError(
                    f"provided `num_downsample` ({num_downsample}) is not "
                    f"identical with `preset_downsample` ({preset_downsample}), "
                    f"please consider set `num_downsample` to {preset_downsample}"
                )
        if increment_config is None:
            increment_config = Preset.increment_configs.get(name)
        # initialization
        super().__init__(-1, in_channels, -1, latent_dim)
        self.to_rgb = Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.net = Backbone(
            name,
            latent_dim=latent_dim,
            pretrained=pretrained,
            need_normalize=need_normalize,
            requires_grad=finetune,
            remove_layers=remove_layers,
            target_layers=target_layers,
            increment_config=increment_config,
            **backbone_kwargs,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.net(self.to_rgb(batch[INPUT_KEY]))


@Encoder1DBase.register("backbone")
class BackboneEncoder1D(Encoder1DBase):
    def __init__(
        self,
        name: str,
        in_channels: int,
        img_size: Optional[int] = None,
        latent_dim: Optional[int] = None,
        *,
        finetune: bool = True,
        pretrained: bool = False,
        need_normalize: bool = False,
        remove_layers: Optional[List[str]] = None,
        target_layers: Optional[Dict[str, str]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        **backbone_kwargs: Any,
    ):
        super().__init__(-1, in_channels, -1)
        self.encoder = BackboneEncoder(
            name,
            in_channels,
            img_size,
            latent_dim,
            finetune=finetune,
            pretrained=pretrained,
            need_normalize=need_normalize,
            remove_layers=remove_layers,
            target_layers=target_layers,
            increment_config=increment_config,
            **backbone_kwargs,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        outputs = self.encoder(batch_idx, batch, state, **kwargs)
        latent = outputs[LATENT_KEY]
        if latent.shape[-2] != 1 or latent.shape[-1] != 1:
            latent = self.pool(latent)
        outputs[LATENT_KEY] = squeeze(latent)
        return outputs


__all__ = [
    "BackboneEncoder",
    "BackboneEncoder1D",
]
