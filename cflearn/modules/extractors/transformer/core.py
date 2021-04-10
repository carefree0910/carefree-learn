import copy
import math
import torch

import numpy as np
import torch.nn as nn

from typing import *
from torch import Tensor
from cftool.misc import register_core

from ..base import ExtractorBase
from ...blocks import _get_norm
from ...blocks import Linear
from ...blocks import Dropout
from ...blocks import PreNorm
from ...blocks import Residual
from ...blocks import Attention
from ...transform.core import Dimensions
from ....misc.toolkit import Activations


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        *,
        dropout: float = 0.0,
        activation: str = "GELU",
        activation_config: Optional[Dict[str, Any]] = None,
        to_latent_config: Optional[Dict[str, Any]] = None,
        from_latent_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim, **(to_latent_config or {})),
            Activations.make(activation, activation_config),
            Dropout(dropout),
            Linear(latent_dim, in_dim, **(from_latent_config or {})),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@Attention.register("decayed")
class DecayedAttention(Attention):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        *,
        seq_len: int,
        dropout: float = 0.0,
        is_self_attention: bool = False,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        activation: Optional[str] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        q_linear_config: Optional[Dict[str, Any]] = None,
        k_linear_config: Optional[Dict[str, Any]] = None,
        v_linear_config: Optional[Dict[str, Any]] = None,
        in_linear_config: Optional[Dict[str, Any]] = None,
        out_linear_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_dim,
            num_heads,
            dropout=dropout,
            is_self_attention=is_self_attention,
            k_dim=k_dim,
            v_dim=v_dim,
            embed_dim=embed_dim,
            activation=activation,
            activation_config=activation_config,
            q_linear_config=q_linear_config,
            k_linear_config=k_linear_config,
            v_linear_config=v_linear_config,
            in_linear_config=in_linear_config,
            out_linear_config=out_linear_config,
        )
        mask = np.zeros([seq_len, seq_len], dtype=np.float32)
        for i in range(1, seq_len):
            np.fill_diagonal(mask[i:], i ** 2)
        mask_ = torch.from_numpy(mask)
        decayed_mask = torch.empty(num_heads, seq_len, seq_len)
        for i in range(num_heads):
            decayed_mask[i] = torch.exp(-(0.1 ** (i + 3)) * mask_)
        self.register_buffer("decayed_mask", decayed_mask)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        last_shapes = weights.shape[1:]
        weights = weights.view(-1, self.num_heads, *last_shapes)
        weights = weights * self.decayed_mask
        weights = weights / (torch.sum(weights, dim=3).unsqueeze(3) + 1.0e-8)
        return weights.view(-1, *last_shapes)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        latent_dim: int = 2048,
        activation: str = "GELU",
        norm_type: str = "layer_norm",
        attention_type: str = "decayed",
        seq_len: Optional[int] = None,
        attention_config: Optional[Dict[str, Any]] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        to_latent_config: Optional[Dict[str, Any]] = None,
        from_latent_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # attention
        if attention_config is None:
            attention_config = {}
        attention_config["is_self_attention"] = True
        attention_config.setdefault("dropout", dropout)
        attn_base = Attention.get(attention_type)
        if attention_type == "decayed":
            if seq_len is None:
                msg = "`seq_len` should be provided when `decayed` attention is used"
                raise ValueError(msg)
            attention_config["seq_len"] = seq_len
        self.attention_block = Residual(
            PreNorm(
                input_dim,
                module=attn_base(input_dim, num_heads, **attention_config),
                norm_type=norm_type,
            )
        )
        # feed forward
        self.feed_forward_block = Residual(
            PreNorm(
                input_dim,
                module=FeedForward(
                    input_dim,
                    latent_dim,
                    dropout=dropout,
                    activation=activation,
                    activation_config=activation_config,
                    to_latent_config=to_latent_config,
                    from_latent_config=from_latent_config,
                ),
                norm_type=norm_type,
            )
        )

    def forward(self, net: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        net = self.attention_block(net, mask=mask)
        net = self.feed_forward_block(net)
        return net


transformer_encoders: Dict[str, Type["TransformerEncoder"]] = {}


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dimensions: Dimensions,
        **kwargs: Any,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.layers = _get_clones(encoder_layer, num_layers)

    def _get_mask(
        self,
        i: int,
        net: Tensor,
        mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        return mask

    def forward(self, net: Tensor, mask: Optional[Tensor]) -> Tensor:
        for i, layer in enumerate(self.layers):
            net = layer(net, mask=self._get_mask(i, net, mask))
        return net

    @classmethod
    def get(cls, name: str) -> Type["TransformerEncoder"]:
        return transformer_encoders[name]

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global transformer_encoders
        return register_core(name, transformer_encoders)


TransformerEncoder.register("basic")(TransformerEncoder)


class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.dropout = Dropout(dropout)
        pe = torch.empty(seq_len, latent_dim)
        position = torch.arange(0, seq_len, dtype=torch.float32)[..., None]
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim)
        )
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe)

    def extra_repr(self) -> str:
        return f"(seq_len): {self.seq_len}"


@ExtractorBase.register("transformer")
class Transformer(ExtractorBase):
    def __init__(
        self,
        in_flat_dim: int,
        dimensions: Dimensions,
        num_heads: int,
        num_layers: int,
        latent_dim: int,
        dropout: float,
        norm_type: str,
        attention_type: str,
        encoder_type: str,
        input_linear_config: Dict[str, Any],
        layer_config: Dict[str, Any],
        encoder_config: Dict[str, Any],
        use_head_token: bool,
        final_norm_type: Optional[str],
    ):
        super().__init__(in_flat_dim, dimensions)
        seq_len = dimensions.num_history
        # latent projection
        self.latent_dim = latent_dim
        self.input_linear = Linear(self.in_dim, latent_dim, **input_linear_config)
        # head token
        if not use_head_token:
            self.head_token = None
        else:
            seq_len += 1
            self.head_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        # position encoding
        self.position_encoding = PositionalEncoding(latent_dim, seq_len, dropout)
        # transformer blocks
        layer_config["dropout"] = dropout
        layer_config["norm_type"] = norm_type
        layer_config["attention_type"] = attention_type
        if attention_type == "decayed":
            layer_config["seq_len"] = seq_len
        layer = TransformerLayer(latent_dim, num_heads, **layer_config)
        encoder_base = TransformerEncoder.get(encoder_type)
        self.encoder = encoder_base(layer, num_layers, dimensions, **encoder_config)
        if final_norm_type is None:
            self.final_norm = None
        else:
            self.final_norm = _get_norm(final_norm_type, latent_dim)

    @property
    def flatten_ts(self) -> bool:
        return False

    @property
    def out_dim(self) -> int:
        return self.latent_dim

    def _aggregate(self, net: Tensor) -> Tensor:
        if self.head_token is not None:
            return net[:, -1]
        return net.mean(1)

    def forward(self, net: Tensor) -> Tensor:
        # input -> latent
        net = self.input_linear(net)
        # concat head token
        if self.head_token is not None:
            expanded_token = self.head_token.expand(net.shape[0], 1, self.latent_dim)
            net = torch.cat([net, expanded_token], dim=1)
        # encode latent vector with transformer
        net = self.position_encoding(net)
        net = self.encoder(net, None)
        # aggregate & norm
        net = self._aggregate(net)
        if self.final_norm is not None:
            net = self.final_norm(net)
        return net


__all__ = ["Transformer"]
