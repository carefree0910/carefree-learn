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
from cftool.misc import safe_execute
from cftool.misc import WithRegister

from .convs import conv_nd
from .convs import Conv2d
from .utils import zero_module
from .common import Lambda
from .hooks import IAttentionHook
from .hijacks import IAttention
from .hijacks import IHijackMixin
from .hijacks import HijackConv2d
from .hijacks import HijackLinear
from .hijacks import HijackCustomLinear
from .activations import Activation
from ...misc.toolkit import sdp_attn
from ...misc.toolkit import gradient_checkpoint


attentions: Dict[str, Type["Attention"]] = {}


class AttentionOutput(NamedTuple):
    output: Tensor
    weights: Optional[Tensor]


class Attention(Module, IAttention, IHijackMixin, WithRegister["Attention"]):
    d = attentions
    customize_sdp: bool = False

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
        hook: Optional[IAttentionHook] = None,
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

        self.out_linear = HijackCustomLinear(
            self.embed_dim,
            input_dim,
            **(out_linear_config or {}),
        )

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

        # hook
        self.hook = hook

    # optional callback, only take effects when `customize_sdp` is set to `True`

    def _get_weights(self, raw_weights: Tensor) -> Tensor:
        # in most cases the softmax version is good enough
        return F.softmax(raw_weights, dim=-1)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        return weights

    # internal

    def _to_heads(self, tensor: Tensor, determinate: bool) -> Tensor:
        seq_len = tensor.shape[1]
        if determinate:
            seq_len = int(seq_len)
        tensor = tensor.view(-1, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

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
        require_weights: bool = False,
        determinate: bool = False,
    ) -> AttentionOutput:
        qkv_inp = q, k, v
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
        if self.hook is not None:
            if self.reduction is not None:
                raise ValueError("currently `hook` does not support `reduction`")
            q, k, v = self.hook.callback(qkv_inp, (q, k, v))
        q, k, v = map(self.activation, [q, k, v])
        # B, N*, D -> B * N_head, N*, D_head
        q, k, v = map(self._to_heads, [q, k, v], [determinate] * 3)
        if mask is not None:
            # B, Nq, Nk -> B, N_head, Nq, Nk
            mask = mask.repeat(self.num_heads, 1, 1)
            mask = mask.view(-1, self.num_heads, *mask.shape[1:])
        if not self.customize_sdp and not require_weights:
            weights = None
            if mask is not None:
                mask = ~mask
            output = sdp_attn(q, k, v, self.training, mask, self.dropout)
        else:
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
        net = self.out_linear(output)
        net = self.activation(net)
        return AttentionOutput(net, weights)


Attention.register("basic")(Attention)


@Attention.register("decayed")
class DecayedAttention(Attention):
    customize_sdp = True

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
        eps: float = 1.0e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.group_norm = nn.GroupNorm(
            num_channels=in_channels,
            num_groups=num_groups,
            eps=eps,
            affine=True,
        )
        self.to_q = HijackConv2d(in_channels, in_channels, 1, 1, 0)
        self.to_k = HijackConv2d(in_channels, in_channels, 1, 1, 0)
        self.to_v = HijackConv2d(in_channels, in_channels, 1, 1, 0)
        self.to_out = HijackConv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.group_norm(net)
        b, c, h, w = net.shape

        q = self.to_q(net)
        k = self.to_k(net)
        v = self.to_v(net)

        area = h * w
        q = q.view(b, c, area).transpose(1, 2)
        k = k.view(b, c, area).transpose(1, 2)
        v = v.view(b, c, area).transpose(1, 2)

        net = sdp_attn(q, k, v, self.training)
        net = net.transpose(1, 2).contiguous().view(b, c, h, w)
        net = self.to_out(net)
        net = inp + net

        return net


class MultiHeadSpatialAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        num_heads: Optional[int] = 1,
        num_head_channels: Optional[int] = None,
        split_qkv_before_heads: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        if num_head_channels is None:
            if num_heads is None:
                msg = "either `num_heads` or `num_head_channels` should be provided"
                raise ValueError(msg)
            self.num_heads = num_heads
        else:
            self.num_heads = in_channels // num_head_channels
        self.split_qkv_before_heads = split_qkv_before_heads
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(32, in_channels)
        self.to_qkv = conv_nd(1, in_channels, in_channels * 3, 1)
        self.to_out = zero_module(conv_nd(1, in_channels, in_channels, 1))

    def forward(self, net: Tensor) -> Tensor:
        return gradient_checkpoint(
            self._forward,
            (net,),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, net: Tensor) -> Tensor:
        b, c, h, w = net.shape
        area = h * w

        inp = net = net.view(b, c, area)
        qkv = self.to_qkv(self.norm(net))
        head_dim = int(c) // self.num_heads
        args = b, c, area, head_dim
        if self.split_qkv_before_heads:
            net = self._split_qkv_before_heads(qkv, *args)
        else:
            net = self._split_qkv_after_heads(qkv, *args)
        net = self.to_out(net)
        return (inp + net).view(b, c, h, w)

    def _split_qkv_before_heads(
        self,
        qkv: Tensor,
        b: int,
        c: int,
        area: int,
        head_dim: int,
    ) -> Tensor:
        q, k, v = qkv.chunk(3, dim=1)

        scale = 1.0 / math.sqrt(math.sqrt(head_dim))
        attn_mat = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(b * self.num_heads, head_dim, area),
            (k * scale).view(b * self.num_heads, head_dim, area),
        )
        attn_prob = F.softmax(attn_mat, dim=-1)
        net = torch.einsum(
            "bts,bcs->bct",
            attn_prob,
            v.contiguous().view(b * self.num_heads, head_dim, area),
        )
        return net.contiguous().view(b, c, area)

    def _split_qkv_after_heads(
        self,
        qkv: Tensor,
        b: int,
        c: int,
        area: int,
        head_dim: int,
    ) -> Tensor:
        qkv = qkv.view(b * self.num_heads, head_dim * 3, area)
        q, k, v = qkv.split(head_dim, dim=1)

        scale = 1.0 / math.sqrt(math.sqrt(head_dim))
        attn_mat = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn_prob = F.softmax(attn_mat, dim=-1)
        net = torch.einsum("bts,bcs->bct", attn_prob, v)
        return net.contiguous().view(b, c, area)


class LinearDepthWiseAttention(Module):
    def __init__(
        self,
        in_channels: int,
        *,
        num_heads: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = num_heads * head_dim
        self.to_qkv = HijackConv2d(in_channels, self.latent_dim * 3, 1, bias=False)
        self.to_out = HijackConv2d(self.latent_dim, in_channels, 1)

    def transpose(self, net: Tensor) -> Tensor:
        # (B, H * W, C) -> (B, H * W, head, dim)
        net = net.view(*net.shape[:-1], self.num_heads, self.head_dim)
        # (B, H * W, head, dim) -> (B, head, dim, H * W)
        net = net.permute(0, 2, 3, 1)
        return net

    def forward(self, net: Tensor) -> Tensor:
        b, c, h, w = net.shape
        qkv = self.to_qkv(net)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self.transpose, [q, k, v])
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        net = torch.einsum("bhde,bhdn->bhen", context, q)
        net = net.contiguous().view(b, self.latent_dim, h, w)
        net = self.to_out(net)
        return net


class CrossAttention(Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        attn_split_chunk: Optional[int] = None,
    ):
        super().__init__()
        self.has_context = context_dim is not None
        latent_dim = head_dim * num_heads
        context_dim = context_dim or query_dim

        self.num_heads = num_heads
        self.attn_split_chunk = attn_split_chunk

        self.to_q = HijackLinear(query_dim, latent_dim, bias=False)
        self.to_k = HijackLinear(context_dim, latent_dim, bias=False)
        self.to_v = HijackLinear(context_dim, latent_dim, bias=False)

        self.out_linear = nn.Sequential(
            HijackLinear(latent_dim, query_dim),
            nn.Dropout(dropout),
        )

    def transpose(self, net: Tensor) -> Tensor:
        b, t, d = net.shape
        dim = d // self.num_heads
        # (B, T, D) -> (B, T, head, dim)
        net = net.view(b, t, self.num_heads, dim)
        # (B, T, head, dim) -> (B, head, T, dim)
        net = net.permute(0, 2, 1, 3)
        # (B, head, T, dim) -> (B * head, T, dim)
        net = net.reshape(b * self.num_heads, t, dim)
        return net

    def forward(
        self,
        net: Tensor,
        *,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # (B, Tq, Dq)
        b, tq, dq = net.shape

        # (B, Tq, D)
        q = self.to_q(net)
        if context is None:
            context = net
        # (B, Tc, D)
        k = self.to_k(context)
        v = self.to_v(context)

        # (B * head, Tq, dim)
        q = self.transpose(q)
        # (B * head, Tc, dim)
        k = self.transpose(k)
        v = self.transpose(v)

        if self.attn_split_chunk is None:
            if mask is not None:
                mask = mask.view(b, -1)
                mask = mask[:, None, :].repeat(self.num_heads, 1, 1)
                mask = ~mask
            net = sdp_attn(q, k, v, self.training, mask)
        else:
            if mask is not None:
                msg = "`mask` is not supported yet when `attn_split_chunk` is enabled"
                raise ValueError(msg)
            size = b * self.num_heads
            # (B * head, Tq, dim)
            net = torch.zeros(size, tq, v.shape[2], dtype=q.dtype, device=q.device)
            for i in range(0, size, self.attn_split_chunk):
                end = i + self.attn_split_chunk
                net[i:end] = sdp_attn(q[i:end], k[i:end], v[i:end], self.training)

        # (B, head, Tq, dim)
        net = net.reshape(b, self.num_heads, tq, dq // self.num_heads)
        # (B, Tq, head, dim)
        net = net.permute(0, 2, 1, 3).contiguous()
        # (B, Tq, D)
        net = net.view(b, tq, dq)
        # (B, Tq, Dq)
        net = self.out_linear(net)
        return net


def make_attention(in_channels: int, attention_type: str, **kwargs: Any) -> Module:
    if attention_type == "none":
        return nn.Identity(in_channels)
    if attention_type == "cross":
        kwargs["query_dim"] = in_channels
        return safe_execute(CrossAttention, kwargs)
    kwargs["in_channels"] = in_channels
    if attention_type in attentions:
        return safe_execute(Attention.get(attention_type), kwargs)
    if attention_type == "spatial":
        return safe_execute(SpatialAttention, kwargs)
    if attention_type == "multi_head_spatial":
        return safe_execute(MultiHeadSpatialAttention, kwargs)
    if attention_type == "linear_depth_wise":
        return safe_execute(LinearDepthWiseAttention, kwargs)
    raise ValueError(f"unrecognized attention type '{attention_type}' occurred")


__all__ = [
    "Attention",
    "DecayedAttention",
    "SpatialAttention",
    "MultiHeadSpatialAttention",
    "LinearDepthWiseAttention",
    "CrossAttention",
    "make_attention",
]
