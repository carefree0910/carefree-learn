import torch

from torch import Tensor

from .common import CommonMLModel
from .common import register_ml_model
from ...schema import forward_results_type
from ...schema import MLConfig


@register_ml_model("wnd")
class WideAndDeepModel(CommonMLModel):
    def build(self, config: MLConfig) -> None:
        self.build_encoder(config)
        encoder = self.encoder
        module_config = config.module_config or {}
        input_dim = module_config["input_dim"]
        if encoder is None or encoder.is_empty:
            wide_dim = deep_dim = input_dim
        else:
            wide_dim = encoder.categorical_dim
            numerical_dim = input_dim - encoder.num_one_hot - encoder.num_embedding
            deep_dim = numerical_dim + encoder.embedding_dim
        module_config["wide_dim"] = wide_dim
        module_config["deep_dim"] = deep_dim
        config.module_config = module_config
        self.build_others(config)

    def forward(self, net: Tensor) -> forward_results_type:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        encoded = self.encode(net)
        one_hot = encoded.one_hot
        embedding = encoded.embedding
        numerical = encoded.numerical
        # wide
        if one_hot is None and embedding is None:
            wide_net = numerical
        else:
            if one_hot is None:
                wide_net = embedding
            elif embedding is None:
                wide_net = one_hot
            else:
                wide_net = torch.cat([one_hot, embedding], dim=-1)
        # deep
        if embedding is None:
            deep_net = numerical
        elif numerical is None:
            deep_net = embedding
        else:
            deep_net = torch.cat([numerical, embedding], dim=-1)
        return self.core(wide_net, deep_net)


__all__ = [
    "WideAndDeepModel",
]
