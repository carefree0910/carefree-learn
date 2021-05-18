import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Optional
from cflearn.types import tensor_dict_type
from cflearn.trainer import TrainerState
from cflearn.constants import PREDICTIONS_KEY
from cflearn.modules.blocks import _get_clones
from cflearn.modules.blocks import Linear
from cflearn.modules.blocks import Dropout
from cflearn.modules.blocks import PreNorm
from cflearn.modules.blocks import Residual
from cflearn.modules.blocks import Attention
from cflearn.modules.blocks import NormFactory
from cflearn.models.ml.protocol import MERGED_KEY
from cflearn.models.ml.protocol import MLCoreProtocol


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            nn.GELU(),
            Dropout(dropout),
            Linear(latent_dim, in_dim),
            Dropout(dropout),
        )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        feed_forward_dim: int,
        *,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.attention = PreNorm(
            dim,
            module=Attention.make(
                "basic",
                config=dict(
                    dim=dim,
                    dropout=dropout,
                    num_heads=num_heads,
                    in_linear_config={"bias": qkv_bias},
                    is_self_attention=True,
                ),
            ),
            norm_type=norm_type,
        )
        self.feed_forward = Residual(
            PreNorm(
                dim,
                module=FeedForward(dim, feed_forward_dim, dropout),
                norm_type=norm_type,
            )
        )

    def forward(self, net: Tensor, return_attention: bool = False) -> Tensor:
        attn = self.attention(net, return_attention=return_attention)
        if return_attention:
            return attn
        net = net + attn
        return self.feed_forward(net)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_history: int,
        feed_forward_dim: int,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        norm_type: str = "layer_norm",
        use_head_token: bool = True,
    ):
        super().__init__()
        self.layers = _get_clones(
            TransformerLayer(
                dim,
                num_heads,
                feed_forward_dim,
                dropout=dropout,
                qkv_bias=qkv_bias,
                norm_type=norm_type,
            ),
            num_layers,
        )
        if not use_head_token:
            self.head_token = None
        else:
            self.head_token = nn.Parameter(torch.zeros(1, 1, dim))
        num_history += int(use_head_token)
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_history, dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.norm = NormFactory(norm_type).make(dim)
        # initializations
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        if self.head_token is not None:
            nn.init.trunc_normal_(self.head_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, net: Tensor) -> Tensor:
        batch_size = net.shape[0]
        if self.head_token is not None:
            head_tokens = self.head_token.repeat([batch_size, 1, 1])
            net = torch.cat([head_tokens, net], dim=1)
        pos_encoding = self.interpolate_pos_encoding(net, self.pos_encoding)
        net = net + pos_encoding
        for layer in self.layers:
            net = layer(net)
        net = self.norm(net)
        if self.head_token is not None:
            return net[:, 0]
        return net.mean(1)

    # this is mainly for vision transformers (ViTs)
    def interpolate_pos_encoding(self, net: Tensor, pos_encoding: Tensor) -> Tensor:
        head_dim = int(self.head_token is not None)
        num_current_history = net.shape[1] - head_dim
        num_history = pos_encoding.shape[1] - head_dim
        if num_current_history == num_history:
            return pos_encoding
        head_encoding = None
        if self.head_token is not None:
            head_encoding = pos_encoding[:, :1]
            pos_encoding = pos_encoding[:, 1:]
        dim = net.shape[-1]
        shape = int(math.sqrt(num_history))
        if shape ** 2 != num_history:
            raise ValueError(f"`num_history` ({num_history}) should be a square number")
        pos_encoding = F.interpolate(
            pos_encoding.reshape(1, shape, shape, dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(num_current_history / num_history),
            mode="bicubic",
        )
        pos_encoding = pos_encoding.permute(0, 2, 3, 1).view(1, -1, dim)
        if head_encoding is None:
            return pos_encoding
        return torch.cat([head_encoding, pos_encoding], dim=1)


@MLCoreProtocol.register("transformer")
class Transformer(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int,
        feed_forward_dim: int,
        *,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        norm_type: str = "batch_norm",
        use_head_token: bool = False,
    ):
        super().__init__(in_dim, out_dim, num_history)
        self.to_latent = Linear(in_dim, latent_dim)
        self.encoder = TransformerEncoder(
            latent_dim,
            num_heads,
            num_history,
            feed_forward_dim,
            num_layers=num_layers,
            dropout=dropout,
            qkv_bias=qkv_bias,
            norm_type=norm_type,
            use_head_token=use_head_token,
        )
        self.head = Linear(latent_dim, out_dim)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[MERGED_KEY]
        net = self.to_latent(net)
        net = self.encoder(net)
        net = self.head(net)
        return {PREDICTIONS_KEY: net}


__all__ = ["Transformer"]
