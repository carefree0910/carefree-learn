import torch

from typing import *
from torch import nn
from torch import Tensor
from cflearn.modules.blocks import BN

from ..base import ExtractorBase
from ...transform.core import Dimensions
from ....misc.toolkit import Activations
from ....modules.blocks import Linear
from ....modules.blocks import Dropout
from ....modules.blocks import Attention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        latent_dim: int = 2048,
        activation: str = "ReLU",
        attention_config: Optional[Dict[str, Any]] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        to_latent_config: Optional[Dict[str, Any]] = None,
        from_latent_config: Optional[Dict[str, Any]] = None,
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

        self.norm1 = BN(input_dim)
        self.norm2 = BN(input_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = Activations.make(activation, activation_config)

    def forward(self, net: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        new = self.self_attn(net, net, net, mask=mask).output
        net = net + self.dropout1(new)
        net = self.norm1(net)
        new = self.from_latent(self.dropout(self.activation(self.to_latent(net))))
        net = net + self.dropout2(new)
        net = self.norm2(net)
        return net


@ExtractorBase.register("transformer")
class Transformer(ExtractorBase):
    def __init__(
        self,
        in_flat_dim: int,
        dimensions: Dimensions,
        num_heads: int,
        num_layers: int,
        latent_dim: int,
        input_linear_config: Dict[str, Any],
        transformer_layer_config: Dict[str, Any],
    ):
        super().__init__(in_flat_dim, dimensions)
        # latent projection
        self.input_linear = Linear(self.in_dim, latent_dim, **input_linear_config)
        self.latent_dim = latent_dim
        # head token
        self.head_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        # position encoding
        pos_shape = 1, dimensions.num_history + 1, latent_dim
        self.position_encoding = nn.Parameter(torch.randn(*pos_shape))
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    latent_dim,
                    num_heads,
                    **transformer_layer_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_bn = BN(latent_dim)

    @property
    def flatten_ts(self) -> bool:
        return False

    @property
    def out_dim(self) -> int:
        return self.latent_dim

    def forward(self, net: Tensor) -> Tensor:
        net = self.input_linear(net)
        expanded_token = self.head_token.expand(net.shape[0], 1, self.latent_dim)
        net = torch.cat([expanded_token, net], dim=1)
        net = net + self.position_encoding
        for layer in self.layers:
            net = layer(net, mask=None)
        return self.final_bn(net[..., 0, :])


__all__ = ["Transformer"]
