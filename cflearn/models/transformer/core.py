import torch

import torch.nn as nn

from typing import Any, Dict
from cfdata.tabular import TabularData

from ...misc.toolkit import *
from ...modules.blocks import *
from ...modules.auxiliary import *
from ..fcnn import FCNN
from ...types import tensor_dict_type


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        attention_config: Dict[str, Any] = None,
        activation: str = "ReLU",
        activation_config: Dict[str, Any] = None,
        latent_dim: int = 2048,
        to_latent_config: Dict[str, Any] = None,
        from_latent_config: Dict[str, Any] = None,
    ):
        super().__init__()
        if attention_config is None:
            attention_config = {}
        attention_config["is_self_attention"] = True
        attention_config.setdefault("dropout", dropout)
        self.self_attn = Attention(input_dim, num_heads, **attention_config)
        if to_latent_config is None:
            to_latent_config = {}
        self.to_latent = Linear(input_dim, latent_dim, **to_latent_config)
        self.dropout = Dropout(dropout)
        if from_latent_config is None:
            from_latent_config = {}
        self.from_latent = Linear(latent_dim, input_dim, **from_latent_config)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = Activations.make(activation, activation_config)

    def forward(self, net: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        new = self.self_attn(net, net, net, mask=mask).output
        net = net + self.dropout1(new)
        net = self.norm1(net)
        new = self.from_latent(self.dropout(self.activation(self.to_latent(net))))
        net = net + self.dropout2(new)
        net = self.norm2(net)
        return net


@FCNN.register("transformer")
class Transformer(FCNN):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super(FCNN, self).__init__(pipeline_config, tr_data, device)
        self._init_fcnn()

    def _init_config(self, tr_data: TabularData) -> None:
        super()._init_config(tr_data)
        transformer_dim = self.tr_data.processed_dim
        transformer_config = self.config.setdefault("transformer_config", {})
        il_config = transformer_config.pop("input_linear_config", None)
        if il_config is None:
            self.input_linear = None
        else:
            il_config.setdefault("bias", False)
            im_latent_dim = il_config.pop("latent_dim", 256)
            self.input_linear = Linear(transformer_dim, im_latent_dim, **il_config)
            transformer_dim = im_latent_dim
        num_layers = transformer_config.pop("num_layers", 6)
        num_heads = transformer_config.pop("num_heads", 8)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    transformer_dim,
                    num_heads,
                    **transformer_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = transformer_config.setdefault("norm", None)
        self.config["fc_in_dim"] = transformer_dim * self.num_history

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch).merge()
        if self.input_linear is not None:
            net = self.input_linear(net)
        mask = batch.get("mask")
        for layer in self.layers:
            net = layer(net, mask=mask)
        if self.norm is not None:
            net = self.norm(net)
        net = self.mlp(net.view(net.shape[0], -1))
        return {"predictions": net}


__all__ = ["Transformer"]
