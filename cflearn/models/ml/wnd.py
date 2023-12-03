import torch

from torch import Tensor
from typing import Any
from typing import Dict

from .common import CommonMLModel
from .common import register_ml_model
from ...schema import forward_results_type


@register_ml_model("wnd")
class WideAndDeepModel(CommonMLModel):
    def mutate_module_config(self, module_config: Dict[str, Any]) -> None:
        encoder = self.encoder
        input_dim = module_config["input_dim"]
        num_history = module_config.get("num_history", 1)
        if encoder is None or encoder.is_empty:
            wide_dim = deep_dim = input_dim
        else:
            wide_dim = encoder.categorical_dim
            numerical_dim = input_dim - encoder.num_one_hot - encoder.num_embedding
            deep_dim = numerical_dim + encoder.embedding_dim
        module_config["wide_dim"] = wide_dim * num_history
        module_config["deep_dim"] = deep_dim * num_history

    def forward(self, net: Tensor) -> forward_results_type:
        encoded = self.encode(net)
        one_hot = encoded.one_hot
        embedding = encoded.embedding
        numerical = encoded.numerical
        # wide
        if one_hot is None and embedding is None:
            assert numerical is not None
            wide_net = numerical
        else:
            if one_hot is None:
                wide_net = embedding
            elif embedding is None:
                wide_net = one_hot
            else:
                wide_net = torch.cat([one_hot, embedding], dim=-1)
        if len(wide_net.shape) > 2:
            wide_net = wide_net.contiguous().view(len(wide_net), -1)
        # deep
        if embedding is None:
            deep_net = numerical if numerical is not None else wide_net
        elif numerical is None:
            deep_net = embedding if embedding is not None else wide_net
        else:
            deep_net = torch.cat([numerical, embedding], dim=-1)
        if len(deep_net.shape) > 2:
            deep_net = deep_net.contiguous().view(len(deep_net), -1)
        return self.core(wide_net, deep_net)


__all__ = [
    "WideAndDeepModel",
]
