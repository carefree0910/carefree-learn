import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..common import CommonDLModel
from ...schema import TDLModel
from ...schema import forward_results_type
from ...schema import to_ml_model
from ...schema import MLConfig
from ...losses import build_loss
from ...modules import build_module
from ...modules.core.ml_encoder import Encoder
from ...modules.core.ml_encoder import MLEncodePack
from ...modules.core.ml_encoder import EncodingResult


def register_ml_model(name: str) -> Callable[[TDLModel], TDLModel]:
    return CommonDLModel.register(to_ml_model(name))


@register_ml_model("common")
class CommonMLModel(CommonDLModel):
    def get_encoder(self) -> Optional[Encoder]:
        return self.m["encoder"]

    def get_module(self) -> nn.Module:
        return self.m["module"]

    def build_encoder(self, config: MLConfig) -> None:
        mapped_encoder_settings = config.mapped_encoder_settings
        if mapped_encoder_settings is None:
            self.m["encoder"] = None
        else:
            self.m["encoder"] = Encoder(
                mapped_encoder_settings,
                config.global_encoder_settings,
            )

    def mutate_module_config(self, module_config: Dict[str, Any]) -> None:
        encoder = self.get_encoder()
        input_dim = module_config["input_dim"]
        num_history = module_config.get("num_history", 1)
        if encoder is not None:
            input_dim += encoder.dim_increment
        input_dim *= num_history
        module_config["input_dim"] = input_dim

    def build_others(self, config: MLConfig, module_config: Dict[str, Any]) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `CommonDLModel`")
        self.m["module"] = build_module(config.module_name, config=module_config)
        self.loss = build_loss(config.loss_name, config=config.loss_config)

    def build(self, config: MLConfig) -> None:
        self.m = nn.ModuleDict()
        self.build_encoder(config)
        module_config = shallow_copy_dict(config.module_config or {})
        self.mutate_module_config(module_config)
        self.build_others(config, module_config)

    def encode(self, net: Tensor) -> MLEncodePack:
        encoder = self.get_encoder()
        if encoder is None or encoder.is_empty:
            return MLEncodePack(None, None, net, None, net)
        numerical_columns = [
            index for index in range(net.shape[-1]) if index not in encoder.tgt_columns
        ]
        numerical = net[..., numerical_columns]
        res: EncodingResult = encoder(net)
        merged_categorical = res.merged
        if merged_categorical is None:
            merged_all = numerical
        else:
            merged_all = torch.cat([numerical, merged_categorical], dim=-1)
        return MLEncodePack(
            res.one_hot,
            res.embedding,
            numerical,
            merged_categorical,
            merged_all,
        )

    def forward(self, net: Tensor, **kwargs: Any) -> forward_results_type:
        net = self.encode(net).merged_all
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.get_module()(net, **kwargs)


@register_ml_model("ml_rnn")
@register_ml_model("ml_fnet")
@register_ml_model("ml_mixer")
@register_ml_model("ml_transformer")
@register_ml_model("ml_pool_former")
class TemporalMLModel(CommonMLModel):
    def mutate_module_config(self, module_config: Dict[str, Any]) -> None:
        encoder = self.get_encoder()
        input_dim = module_config["input_dim"]
        if encoder is not None:
            input_dim += encoder.dim_increment
        module_config["input_dim"] = input_dim

    def forward(self, net: Tensor, **kwargs: Any) -> forward_results_type:
        net = self.encode(net).merged_all
        return self.get_module()(net, **kwargs)


__all__ = [
    "register_ml_model",
    "CommonMLModel",
    "TemporalMLModel",
]
