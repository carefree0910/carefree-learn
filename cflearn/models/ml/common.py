import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..common import CommonDLModel
from ...schema import forward_results_type
from ...schema import to_ml_model
from ...schema import MLConfig
from ...losses import build_loss
from ...modules import build_ml_module
from ...modules.core.ml_encoder import Encoder
from ...modules.core.ml_encoder import MLEncodePack
from ...modules.core.ml_encoder import EncodingResult


def register_ml_model(name: str) -> Callable:
    return CommonDLModel.register(to_ml_model(name))


@register_ml_model("common")
class CommonMLModel(CommonDLModel):
    encoder: Optional[Encoder]

    def build_encoder(self, config: MLConfig) -> None:
        mapped_encoder_settings = config.mapped_encoder_settings
        if mapped_encoder_settings is None:
            self.encoder = None
        else:
            self.encoder = Encoder(
                mapped_encoder_settings,
                config.global_encoder_settings,
            )

    def mutate_module_config(self, module_config: Dict[str, Any]) -> None:
        input_dim = module_config["input_dim"]
        num_history = module_config.get("num_history", 1)
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        input_dim *= num_history
        module_config["input_dim"] = input_dim

    def build_others(self, config: MLConfig, module_config: Dict[str, Any]) -> None:
        if config.loss_name is None:
            raise ValueError("`loss_name` should be specified for `CommonDLModel`")
        self.core = build_ml_module(config.module_name, config=module_config)
        self.m = nn.ModuleDict({"module": self.core, "encoder": self.encoder})
        self.loss = build_loss(config.loss_name, config=config.loss_config)

    def build(self, config: MLConfig) -> None:
        self.build_encoder(config)
        module_config = shallow_copy_dict(config.module_config or {})
        self.mutate_module_config(module_config)
        self.build_others(config, module_config)

    def encode(self, net: Tensor) -> MLEncodePack:
        if self.encoder is None or self.encoder.is_empty:
            return MLEncodePack(None, None, net, None, net)
        numerical_columns = [
            index
            for index in range(net.shape[-1])
            if index not in self.encoder.tgt_columns
        ]
        numerical = net[..., numerical_columns]
        res: EncodingResult = self.encoder(net)
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

    def forward(self, net: Tensor) -> forward_results_type:
        net = self.encode(net).merged_all
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.core(net)


__all__ = [
    "CommonMLModel",
]
