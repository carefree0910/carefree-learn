import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Optional
from typing import NamedTuple
from functools import partial
from torch.nn import Module
from cftool.misc import WithRegister

from .convs import Conv2d
from .common import Lambda
from .customs import Linear
from .activations import Activation


attentions: Dict[str, Type["Attention"]] = {}


class AttentionOutput(NamedTuple):
    output: Tensor
    weights: Tensor


class Attention(Module, WithRegister["Attention"]):
    d = attentions

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        qk_scale: Optional[float] = None,
        kv_same: Optional[bool] = None,
        qkv_bias_same: bool = True,
        is_self_attention: bool = False,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        activation: Optional[str] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        out_linear_config: Optional[Dict[str, Any]] = None,
        reduction_ratio: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        if kv_same is None:
            if k_dim is not None and v_dim is not None and k_dim != v_dim:
                kv_same = False
            else:
                kv_same = True
        self.kv_same = kv_same
        has_reduction = reduction_ratio is not None and reduction_ratio > 1
        self.qkv_same = is_self_attention and not has_reduction
        if not is_self_attention:
            self.k_dim = k_dim or input_dim
            self.v_dim = v_dim or self.k_dim
        else:
            if k_dim is not None and k_dim != input_dim:
                raise ValueError("self attention is used but `k_dim` != `input_dim`")
            if v_dim is not None and v_dim != input_dim:
                raise ValueError("self attention is used but `v_dim` != `input_dim`")
            self.k_dim = self.v_dim = input_dim
        self.embed_dim = embed_dim or input_dim

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scaling = qk_scale or float(self.head_dim) ** 0.5
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("`embed_dim` must be divisible by `num_heads`")

        if self.qkv_same:
            self.kv_w = self.q_w = self.k_w = self.v_w = None
            self.in_w = nn.Parameter(torch.empty(3 * self.embed_dim, input_dim))
            nn.init.trunc_normal_(self.in_w, std=0.02)
        elif kv_same:
            self.in_w = self.k_w = self.v_w = None
            self.q_w = nn.Parameter(torch.empty(self.embed_dim, input_dim))
            self.kv_w = nn.Parameter(torch.empty(2 * self.embed_dim, input_dim))
            nn.init.trunc_normal_(self.q_w, std=0.02)
            nn.init.trunc_normal_(self.kv_w, std=0.02)
        else:
            self.in_w = None
            self.q_w = nn.Parameter(torch.empty(self.embed_dim, input_dim))
            self.k_w = nn.Parameter(torch.empty(self.embed_dim, self.k_dim))
            self.v_w = nn.Parameter(torch.empty(self.embed_dim, self.v_dim))
            nn.init.xavier_uniform_(self.q_w)
            nn.init.xavier_uniform_(self.k_w)
            nn.init.xavier_uniform_(self.v_w)
        if not bias:
            self.q_bias = self.k_bias = self.v_bias = None
            self.kv_bias = self.qkv_bias = None
        elif not qkv_bias_same:
            self.kv_bias = self.qkv_bias = None
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.k_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        elif self.qkv_same or not kv_same:
            self.q_bias = self.k_bias = self.v_bias = self.kv_bias = None
            self.qkv_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))
        else:
            self.k_bias = self.v_bias = self.qkv_bias = None
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.kv_bias = nn.Parameter(torch.zeros(2 * self.embed_dim))

        if out_linear_config is None:
            out_linear_config = {}
        self.out_linear = Linear(self.embed_dim, input_dim, **out_linear_config)

        self.dropout = dropout
        self.activation = Activation.make(activation, activation_config)

        if not has_reduction:
            self.reduction = None
        else:
            self.reduction = nn.Sequential(
                Conv2d(
                    self.embed_dim,
                    self.embed_dim,
                    kernel_size=reduction_ratio,  # type: ignore
                    stride=reduction_ratio,  # type: ignore
                    padding=0,
                ),
                Lambda(lambda t: t.flatten(2).transpose(1, 2)),
                nn.LayerNorm(self.embed_dim),
            )

    def _to_heads(self, tensor: Tensor, determinate: bool) -> Tensor:
        seq_len = tensor.shape[1]
        if determinate:
            seq_len = int(seq_len)
        tensor = tensor.view(-1, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def _get_weights(self, raw_weights: Tensor) -> Tensor:
        # in most cases the softmax version is good enough
        return F.softmax(raw_weights, dim=-1)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        return weights

    def _reduce(self, net: Tensor, hw: Optional[Tuple[int, int]] = None) -> Tensor:
        if self.reduction is None:
            return net
        if hw is None:
            msg = "`hw` should be provided when `reduction` is applied"
            raise ValueError(msg)
        net = net.transpose(1, 2).contiguous()
        net = net.view(-1, net.shape[1], *hw)
        net = self.reduction(net)
        return net

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        hw: Optional[Tuple[int, int]] = None,
        mask: Optional[Tensor] = None,
        determinate: bool = False,
    ) -> AttentionOutput:
        # `mask` represents slots which will be zeroed
        if self.qkv_same:
            qkv = F.linear(q, self.in_w, self.qkv_bias)
            if not determinate:
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                qkv = qkv.view(-1, int(q.shape[1]), 3, self.embed_dim)
                q, k, v = map(partial(torch.squeeze, dim=2), qkv.split(1, dim=2))
        elif self.kv_same:
            # B, Nq, Din -> B, Nq, D
            q = F.linear(q, self.q_w, self.q_bias)
            # B, Nk, Dk -> B, Nk, D
            if self.reduction is not None:
                if hw is None:
                    msg = "`hw` should be provided when `reduction` is applied"
                    raise ValueError(msg)
                k = self._reduce(k, hw)
            k, v = F.linear(k, self.kv_w, self.kv_bias).chunk(2, dim=-1)
        else:
            if self.qkv_bias is not None:
                q_bias, k_bias, v_bias = self.qkv_bias.chunk(3)
            else:
                q_bias = self.q_bias
                k_bias = self.k_bias
                v_bias = self.v_bias
            # B, Nq, Din -> B, Nq, D
            q = F.linear(q, self.q_w, q_bias)
            # B, Nk, Dk -> B, Nk, D
            k = F.linear(self._reduce(k, hw), self.k_w, k_bias)
            # B, Nv, Dv -> B, Nv, D
            v = F.linear(self._reduce(v, hw), self.v_w, v_bias)
        q, k, v = map(self.activation, [q, k, v])
        # B, N*, D -> B * N_head, N*, D_head
        q, k, v = map(self._to_heads, [q, k, v], [determinate] * 3)
        if mask is not None:
            # B, Nq, Nk -> B, N_head, Nq, Nk
            mask = mask.repeat(self.num_heads, 1, 1)
            mask = mask.view(-1, self.num_heads, *mask.shape[1:])
        # B, N_head, Nq, Nk
        raw_weights = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            raw_weights.masked_fill_(mask, float("-inf"))
        # scale
        raw_weights = raw_weights / self.scaling
        # B, N_head, Nq, Nk -> B, N_head, Nq, Nk
        weights = self._get_weights(raw_weights)
        if 0.0 < self.dropout < 1.0:
            weights = F.dropout(weights, self.dropout, self.training)
        weights = self._weights_callback(weights)
        # B, N_head, Nq, D_head
        output = torch.matmul(weights, v)
        # B, N_head, Nq, D_head -> B, Nq, N_head, D_head
        output = output.transpose(1, 2).contiguous()
        # B, Nq, N_head, D_head -> B, Nq, D
        seq_len = output.shape[1]
        if determinate:
            seq_len = int(seq_len)
        output = output.view(-1, seq_len, self.embed_dim)
        # B, Nq, D -> B, Nq, Din
        output = self.activation(self.out_linear(output))
        return AttentionOutput(output, weights)


Attention.register("basic")(Attention)


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
            out_linear_config=out_linear_config,
        )
        mask = np.zeros([seq_len, seq_len], dtype=np.float32)
        for i in range(1, seq_len):
            np.fill_diagonal(mask[i:], i**2)
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


class SpatialAttention(Module):
    def __init__(
        self,
        in_channels: int,
        *,
        num_groups: int = 32,
        num_head_channels: Optional[int] = None,
        rescale_output_factor: float = 1.0,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_head_channels = num_head_channels
        if num_head_channels is None:
            self.num_heads = 1
        else:
            self.num_heads = in_channels // num_head_channels
        self.group_norm = nn.GroupNorm(
            num_channels=in_channels,
            num_groups=num_groups,
            eps=eps,
            affine=True,
        )

        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)

        self.rescale_output_factor = rescale_output_factor
        self.out_linear = nn.Linear(in_channels, in_channels)

    def transpose(self, net: Tensor) -> Tensor:
        # (B, H * W, C) -> (B, H * W, head, dim)
        net = net.view(*net.shape[:-1], self.num_heads, -1)
        # (B, H * W, head, dim) -> (B, head, H * W, dim)
        net = net.permute(0, 2, 1, 3)
        return net

    def forward(self, net: Tensor) -> Tensor:
        # (B, C, H, W)
        inp = net
        batch, channel, height, width = net.shape
        net = self.group_norm(net)
        # (B, C, H * W)
        net = net.view(batch, channel, height * width)
        # (B, H * W, C)
        net = net.transpose(1, 2)

        # (B, H * W, C)
        q_net = self.to_q(net)
        k_net = self.to_k(net)
        v_net = self.to_v(net)

        # (B, head, H * W, dim)
        q_net = self.transpose(q_net)
        k_net = self.transpose(k_net)
        v_net = self.transpose(v_net)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.in_channels / self.num_heads))
        # (B, head, H * W, H * W)
        attn_mat = torch.matmul(q_net * scale, k_net.transpose(-1, -2) * scale)
        attn_prob = torch.softmax(attn_mat.float(), dim=-1).to(attn_mat.dtype)

        # (B, head, H * W, dim)
        net = torch.matmul(attn_prob, v_net)
        # (B, H * W, head, dim)
        net = net.permute(0, 2, 1, 3).contiguous()
        # (B, H * W, C)
        net = net.view(*net.shape[:-2], self.in_channels)
        # (B, H * W, C)
        net = self.out_linear(net)
        # (B, C, H, W)
        net = net.transpose(-1, -2).reshape(batch, channel, height, width)

        # (B, C, H, W)
        net = (net + inp) / self.rescale_output_factor
        return net


__all__ = [
    "Attention",
    "DecayedAttention",
    "SpatialAttention",
]
