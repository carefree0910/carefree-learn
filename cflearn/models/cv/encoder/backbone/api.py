from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional

from .core import Backbone
from ..protocol import Encoder1DBase
from .....types import tensor_dict_type
from .....trainer import TrainerState
from .....constants import INPUT_KEY
from .....constants import WARNING_PREFIX
from .....modules.blocks import Conv2d


class Preset:
    remove_layers: Dict[str, List[str]] = {}
    target_layers: Dict[str, Dict[str, str]] = {}
    latent_dims: Dict[str, int] = {}
    increment_configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_settings(cls) -> Callable:
        def _register(settings: Preset) -> None:
            cls.remove_layers.update(settings.remove_layers)
            cls.target_layers.update(settings.target_layers)
            cls.latent_dims.update(settings.latent_dims)
            cls.increment_configs.update(settings.increment_configs)

        return _register


@Encoder1DBase.register("backbone")
class BackboneEncoder(Encoder1DBase):
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        in_channels: int = 3,
        img_size: Optional[int] = None,
        latent_dim: Optional[int] = None,
        *,
        finetune: bool = True,
        pretrained: bool = False,
        need_normalize: bool = False,
        remove_layers: Optional[List[str]] = None,
        target_layers: Optional[Dict[str, str]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(img_size, in_channels, latent_dim)
        if img_size is not None:
            print(f"{WARNING_PREFIX}`img_size` will not affect `BackboneEncoder`")
        self.to_rgb = None
        if in_channels != 3:
            self.to_rgb = Conv2d(in_channels, 3, kernel_size=1, bias=False)
        if config is None:
            config: Dict[str, Any] = {}
        remove_layers = config.setdefault("backbone_remove_layers", remove_layers)
        target_layers = config.setdefault("backbone_target_layers", target_layers)
        latent_dim = config.get("backbone_latent_dim", latent_dim)
        increment_config = config.setdefault("backbone_increment_configs", increment_config)
        if remove_layers is None:
            remove_layers = Preset.remove_layers.get(name)
            if remove_layers is None:
                msg = f"`remove_layers` should be provided for `{name}`"
                raise ValueError(msg)
            config["backbone_remove_layers"] = remove_layers
        if target_layers is None:
            target_layers = Preset.target_layers.get(name)
            if target_layers is None:
                msg = f"`target_layers` should be provided for `{name}`"
                raise ValueError(msg)
            config["backbone_target_layers"] = target_layers
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
        if increment_config is None:
            increment_config = Preset.increment_configs.get(name)
        config.setdefault("backbone_finetune", finetune)
        config.setdefault("backbone_pretrained", pretrained)
        config.setdefault("backbone_need_normalize", need_normalize)
        self.net = Backbone(
            name,
            latent_dim=latent_dim,
            pretrained=config["backbone_pretrained"],
            need_normalize=config["backbone_need_normalize"],
            requires_grad=config["backbone_finetune"],
            remove_layers=config["backbone_remove_layers"],
            target_layers=config["backbone_target_layers"],
            increment_config=increment_config,
            **config.setdefault("backbone_kwargs", {}),
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        if self.to_rgb is not None:
            net = self.to_rgb(net)
        return self.net(net)


__all__ = ["BackboneEncoder"]
