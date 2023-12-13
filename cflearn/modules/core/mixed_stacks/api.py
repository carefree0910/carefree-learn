import math
import torch

import numpy as np
import torch.nn as nn

from enum import Enum
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from typing import Protocol
from dataclasses import field
from dataclasses import dataclass
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import shallow_copy_dict

from .common import build_token_mixer
from .common import build_channel_mixer
from .poolers import BertPooler
from .poolers import SequencePooler
from .channel_mixers import FeedForward
from ..norms import NormFactory
from ..customs import Linear
from ..customs import DropPath
from ..hijacks import HijackConv2d
from ..hijacks import HijackLinear
from ..high_level import PreNorm
from ..attentions import CrossAttention
from ...common import zero_module
from ...common import Lambda
from ....toolkit import new_seed
from ....toolkit import interpolate
from ....toolkit import gradient_checkpoint


class MixingBlock(Module):
    def __init__(
        self,
        layer_idx: int,
        num_layers: int,
        num_tokens: int,
        in_dim: int,
        latent_dim: int,
        *,
        norm_position: str = "pre_norm",
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        token_mixing_dropout: Optional[float] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        residual_after_norm: bool = False,
    ):
        def _make_norm() -> nn.Module:
            factory = NormFactory(norm_type)
            return factory.make(in_dim, **(norm_kwargs or {}))

        super().__init__()
        if norm_position not in {"pre_norm", "post_norm"}:
            raise ValueError(
                "`norm_position` should be either 'pre_norm' or 'post_norm', "
                f"'{norm_position}' found"
            )
        self.norm_position = norm_position
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # token mixing
        if token_mixing_config is None:
            token_mixing_config = {}
        token_mixing_config.update(
            {
                "layer_idx": layer_idx,
                "num_layers": num_layers,
                "num_tokens": num_tokens,
                "in_dim": in_dim,
                "latent_dim": latent_dim,
                "dropout": dropout,
            }
        )
        self.token_norm = _make_norm()
        self.token_mixing = build_token_mixer(
            token_mixing_type, config=token_mixing_config
        )
        if token_mixing_dropout is None:
            token_mixing_dropout = dropout
        self.token_mixing_dropout = nn.Dropout(token_mixing_dropout)
        # channel mixing
        if channel_mixing_config is None:
            channel_mixing_config = {}
        channel_mixing_config.update(
            {
                "layer_idx": layer_idx,
                "num_layers": num_layers,
                "in_dim": in_dim,
                "latent_dim": latent_dim,
                "dropout": dropout,
            }
        )
        self.residual_after_norm = residual_after_norm
        self.channel_norm = _make_norm()
        self.channel_mixing = build_channel_mixer(
            channel_mixing_type, config=channel_mixing_config
        )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if self.norm_position == "pre_norm":
            return self._pre_norm_forward(
                net, hw, deterministic=deterministic, **kwargs
            )
        if self.norm_position == "post_norm":
            return self._post_norm_forward(
                net, hw, deterministic=deterministic, **kwargs
            )
        raise ValueError(f"unrecognized norm_position '{self.norm_position}' occurred")

    def _pre_norm_forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        # token mixing
        token_mixing_kw = dict(hw=hw, deterministic=deterministic)
        token_mixing_kw.update(kwargs)
        token_mixing_net = self.token_norm(net)
        token_mixing_net = self.token_mixing(token_mixing_net, **token_mixing_kw)
        token_mixing_net = self.token_mixing_dropout(token_mixing_net)
        net = net + self.drop_path(token_mixing_net)
        # channel mixing
        if self.residual_after_norm:
            net = self.channel_norm(net)
        if not self.channel_mixing.need_2d:
            channel_mixing_net = net
        else:
            if hw is None:
                raise ValueError("`hw` should be provided when FFN needs 2d input")
            channel_mixing_net = net.view(-1, *hw, net.shape[-1])
        if not self.residual_after_norm:
            channel_mixing_net = self.channel_norm(channel_mixing_net)
        channel_mixing_net = self.channel_mixing(channel_mixing_net)
        net = net + self.drop_path(channel_mixing_net)
        return net

    def _post_norm_forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        # token mixing
        token_mixing_kw = dict(hw=hw, deterministic=deterministic)
        token_mixing_kw.update(kwargs)
        token_mixing_net = self.token_mixing(net, **token_mixing_kw)
        token_mixing_net = self.token_mixing_dropout(token_mixing_net)
        net = net + self.drop_path(token_mixing_net)
        net = self.token_norm(net)
        # channel mixing
        if not self.channel_mixing.need_2d:
            channel_mixing_net = net
        else:
            if hw is None:
                raise ValueError("`hw` should be provided when FFN needs 2d input")
            channel_mixing_net = net.view(-1, *hw, net.shape[-1])
        channel_mixing_net = self.channel_mixing(channel_mixing_net)
        net = net + self.drop_path(channel_mixing_net)
        net = self.channel_norm(net)
        return net


class PositionalEncoding(Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        dropout: float = 0.0,
        *,
        num_head_tokens: int,
        is_vision: bool,
        enable: bool = True,
    ):
        super().__init__()
        self.pos_drop = None
        self.pos_encoding = None
        if enable:
            self.pos_drop = nn.Dropout(p=dropout)
            self.pos_encoding = nn.Parameter(torch.zeros(1, num_tokens, dim))
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self.num_head_tokens = num_head_tokens
        self.is_vision = is_vision

    def forward(
        self,
        net: Tensor,
        *,
        hwp: Optional[Tuple[int, int, int]] = None,
    ) -> Tensor:
        if self.pos_encoding is None or self.pos_drop is None:
            return net
        if self.is_vision:
            feature_span = 0
            pos_encoding = self.interpolate_pos_encoding(net, hwp)
        else:
            feature_span = net.shape[1] - self.num_head_tokens
            pos_encoding = self.pos_encoding[:, :feature_span]
        pos_encoding = self.pos_drop(pos_encoding)
        if self.is_vision or self.num_head_tokens <= 0:
            return net + pos_encoding
        net = net.clone()
        net[:, :feature_span] += pos_encoding
        return net

    # this is for vision positional encodings
    def interpolate_pos_encoding(
        self,
        net: Tensor,
        hwp: Optional[Tuple[int, int, int]],
    ) -> Tensor:
        pos_encoding = self.pos_encoding
        assert pos_encoding is not None
        num_current_history = net.shape[1] - self.num_head_tokens
        num_history = pos_encoding.shape[1] - self.num_head_tokens
        if hwp is None:
            w = h = patch_size = None
        else:
            h, w, patch_size = hwp
        if num_current_history == num_history and w == h:
            return pos_encoding
        if w is None or h is None or patch_size is None:
            raise ValueError("`hwp` should be provided for `interpolate_pos_encoding`")
        head_encoding = None
        if self.num_head_tokens > 0:
            head_encoding = pos_encoding[:, : self.num_head_tokens]
            pos_encoding = pos_encoding[:, self.num_head_tokens :]
        dim = net.shape[-1]
        # This assume that the original input is squared image
        sqrt = math.sqrt(num_history)
        wh_ratio = w / h
        pw = math.sqrt(num_current_history * wh_ratio) + 0.1
        ph = math.sqrt(num_current_history / wh_ratio) + 0.1
        pos_encoding = interpolate(
            pos_encoding.reshape(1, int(sqrt), int(sqrt), dim).permute(0, 3, 1, 2),
            factor=(pw / sqrt, ph / sqrt),
            mode="bicubic",
        )
        assert int(pw) == pos_encoding.shape[-2] and int(ph) == pos_encoding.shape[-1]
        pos_encoding = pos_encoding.permute(0, 2, 3, 1).view(1, -1, dim)
        if head_encoding is None:
            return pos_encoding
        return torch.cat([head_encoding, pos_encoding], dim=1)


class MixedStackedEncoder(Module):
    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        *,
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 4,
        dropout: float = 0.0,
        dpr_list: Optional[List[float]] = None,
        drop_path_rate: float = 0.1,
        norm_position: str = "pre_norm",
        norm_type: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        embedding_norm: Optional[nn.Module] = None,
        embedding_dropout: Optional[float] = None,
        residual_after_norm: bool = False,
        latent_dim: Optional[int] = None,
        latent_dim_ratio: float = 1.0,
        use_head_token: bool = False,
        head_pooler: Optional[str] = "mean",
        use_positional_encoding: bool = False,
        is_vision_positional_encoding: Optional[bool] = None,
        positional_encoding_dropout: float = 0.0,
        no_head_norm: Optional[bool] = None,
        norm_after_head: bool = False,
        aux_heads: Optional[List[str]] = None,
    ):
        super().__init__()
        # head token
        self.aux_heads = aux_heads
        if not use_head_token:
            num_head_tokens = 0
            self.head_token = None
        else:
            num_head_tokens = 1
            if aux_heads is not None:
                num_head_tokens += len(aux_heads)
            self.head_token = nn.Parameter(torch.zeros(1, num_head_tokens, in_dim))
        self.num_heads = num_head_tokens
        # positional encoding
        num_tokens += num_head_tokens
        if is_vision_positional_encoding is None:
            if use_positional_encoding:
                raise ValueError(
                    "`is_vision_positional_encoding` should be specified when "
                    "`use_positional_encoding` is set to True"
                )
            is_vision_positional_encoding = False
        self.pos_encoding = PositionalEncoding(
            in_dim,
            num_tokens,
            positional_encoding_dropout,
            num_head_tokens=num_head_tokens,
            is_vision=is_vision_positional_encoding,
            enable=use_positional_encoding,
        )
        self.embedding_norm = embedding_norm
        if embedding_dropout is None:
            self.embedding_dropout = None
        else:
            self.embedding_dropout = nn.Dropout(embedding_dropout)
        # core
        if dpr_list is None:
            dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        if latent_dim is None:
            latent_dim = int(round(in_dim * latent_dim_ratio))
        self.mixing_blocks = ModuleList(
            [
                MixingBlock(
                    i,
                    num_layers,
                    num_tokens,
                    in_dim,
                    latent_dim,
                    norm_position=norm_position,
                    token_mixing_type=token_mixing_type,
                    token_mixing_config=token_mixing_config,
                    channel_mixing_type=channel_mixing_type,
                    channel_mixing_config=channel_mixing_config,
                    dropout=dropout,
                    drop_path=drop_path,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    residual_after_norm=residual_after_norm,
                )
                for i, drop_path in enumerate(dpr_list)
            ]
        )
        # head
        if self.head_token is not None:
            if self.aux_heads is None:
                head = Lambda(lambda x: x[:, 0], name="head_token")
            else:
                head = Lambda(lambda x: x[:, : self.num_heads], name="head_token")
        else:
            if head_pooler is None:
                head = nn.Identity()
            elif head_pooler == "sequence":
                head = SequencePooler(in_dim, aux_heads)
            else:
                if aux_heads is not None:
                    raise ValueError(
                        "either `head_token` or `sequence` head_pooler should be used "
                        f"when `aux_heads` ({aux_heads}) is provided"
                    )
                if head_pooler == "bert":
                    head = BertPooler(in_dim)
                elif head_pooler == "mean":
                    head = Lambda(lambda x: x.mean(1), name="global_average")
                else:
                    msg = f"unrecognized head_pooler '{head_pooler}' occurred"
                    raise ValueError(msg)
        # head norm
        if no_head_norm is None:
            no_head_norm = norm_position == "post_norm"
        if no_head_norm:
            self.head_norm = None
            self.head = head
        elif norm_after_head:
            self.head_norm = NormFactory(norm_type).make(in_dim, **(norm_kwargs or {}))
            self.head = head
        else:
            self.head_norm = None
            self.head = PreNorm(
                in_dim,
                module=head,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
            )
        # initializations
        if self.head_token is not None:
            nn.init.trunc_normal_(self.head_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, Linear):
            m.init_weights_with(lambda t: nn.init.trunc_normal_(t, std=0.02))
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def pre_process(
        self,
        net: Tensor,
        *,
        hwp: Optional[Tuple[int, int, int]] = None,
        deterministic: bool = False,
    ) -> Tensor:
        n, t, d = net.shape
        if self.head_token is not None:
            head_tokens = self.head_token.repeat([n, 1, 1])
            net = torch.cat([head_tokens, net], dim=1)
            t += 1
        if deterministic:
            net = net.view(-1, *map(int, [t, d]))
        net = self.pos_encoding(net, hwp=hwp)
        if self.embedding_norm is not None:
            net = self.embedding_norm(net)
        if self.embedding_dropout is not None:
            net = self.embedding_dropout(net)
        return net

    def post_process(self, net: Tensor) -> Tensor:
        net = self.head(net)
        if self.head_norm is not None:
            net = self.head_norm(net)
        return net

    def forward(
        self,
        net: Tensor,
        *,
        hw: Optional[Tuple[int, int]] = None,
        hwp: Optional[Tuple[int, int, int]] = None,
        deterministic: bool = False,
    ) -> Tensor:
        net = self.pre_process(net, hwp=hwp, deterministic=deterministic)
        for block in self.mixing_blocks:
            net = block(net, hw, deterministic=deterministic)
        net = self.post_process(net)
        return net


def do_nothing(x: Tensor) -> Tensor:
    return x


def gather(net: Tensor, dim: int, index: Tensor) -> Tensor:
    if net.device.type != "mps" or net.shape[-1] != 1:
        return torch.gather(net, dim, index)
    if dim < 0:
        dim = dim - 1
    return torch.gather(net.unsqueeze(-1), dim, index.unsqueeze(-1)).squeeze(-1)


def bipartite_soft_matching_random2d(
    metric: Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    seed: int,
    no_rand: bool = False,
) -> Tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        device = metric.device
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=device, dtype=torch.int64)
        else:
            generator = torch.Generator(device).manual_seed(seed)
            rand_idx = torch.randint(
                sy * sx, size=(hsy, wsx, 1), generator=generator, device=device
            )

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy * sx, device=device, dtype=torch.int64
        )
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx))
        idx_buffer_view = (
            idx_buffer_view.view(hsy, wsx, sy, sx)
            .transpose(1, 2)
            .reshape(hsy * sy, wsx * sx)
        )

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=device, dtype=torch.int64)
            idx_buffer[: (hsy * sy), : (wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x: Tensor) -> Tuple[Tensor, Tensor]:
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: Tensor) -> Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce="mean")

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: Tensor) -> Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, r, c),
            src=src,
        )

        return out

    return merge, unmerge


def compute_merge(x: Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    if downsample > tome_info["max_downsample"]:
        m, u = do_nothing, do_nothing
    else:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * tome_info["ratio"])
        m, u = bipartite_soft_matching_random2d(  # type: ignore
            x,
            w,
            h,
            tome_info["sx"],
            tome_info["sy"],
            r,
            tome_info.get("seed", new_seed()),
            not tome_info["use_rand"],
        )

    m_a, u_a = (m, u) if tome_info["merge_attn"] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if tome_info["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if tome_info["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


class ReferenceMode(str, Enum):
    READ = "read"
    WRITE = "write"


class GetRefNet(Protocol):
    def __call__(self, timesteps: Tensor) -> Tensor:
        pass


@dataclass
class StyleReferenceStates:
    style_fidelity: float = 0.5
    reference_weight: float = 1.0
    get_ref_net: Optional[GetRefNet] = None
    # runtime states
    mode: ReferenceMode = ReferenceMode.READ
    bank: List[Tensor] = field(default_factory=lambda: [])
    in_write: bool = False
    is_written: bool = False
    enable_write: bool = False
    uncond_indices: Optional[List[int]] = None


class SpatialTransformerHooks:
    def __init__(self, m: "SpatialTransformerBlock"):
        self.m = m
        self.setup()

    @property
    def enabled(self) -> bool:
        return self.tome_info is not None or self.style_reference_states is not None

    def setup(
        self,
        *,
        tome_info: Optional[Dict[str, Any]] = None,
        style_reference_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        # tomesd (https://github.com/dbolya/tomesd)
        if tome_info is None:
            self.tome_info = None
        else:
            self.tome_info = shallow_copy_dict(tome_info)
            self.tome_info.setdefault("ratio", 0.5)
            self.tome_info.setdefault("max_downsample", 1)
            self.tome_info.setdefault("sx", 2)
            self.tome_info.setdefault("sy", 2)
            self.tome_info.setdefault("use_rand", True)
            self.tome_info.setdefault("merge_attn", True)
            self.tome_info.setdefault("merge_crossattn", False)
            self.tome_info.setdefault("merge_mlp", False)
        # style reference
        if style_reference_states is None:
            self.style_reference_states = None
        else:
            self.style_reference_states = StyleReferenceStates(**style_reference_states)

    def forward(self, net: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.tome_info is not None:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(net, self.tome_info)
            net = u_a(self.m.attn1(m_a(self.m.norm1(net)))) + net
            net = u_c(self.m.attn2(m_c(self.m.norm2(net)), context=context)) + net
            net = u_m(self.m.ff(m_m(self.m.norm3(net)))) + net
            return net
        if self.style_reference_states is not None:
            net_norm1 = self.m.norm1(net)
            net_attn1 = None
            if self.style_reference_states.mode == ReferenceMode.WRITE:
                if self.style_reference_states.enable_write:
                    self.style_reference_states.bank.append(net_norm1.detach().clone())
                self.style_reference_states.is_written = True
            elif self.style_reference_states.mode == ReferenceMode.READ:
                bank = self.style_reference_states.bank
                if len(bank) > 0:
                    bank_ctx = torch.cat([net_norm1] + bank, dim=1)
                    net_attn1 = self.m.attn1(net_norm1, context=bank_ctx)
                    style_cfg = self.style_reference_states.style_fidelity
                    uc_indices = self.style_reference_states.uncond_indices
                    if uc_indices and style_cfg > 1.0e-5:
                        net_attn1_original_uc = net_attn1.clone()
                        net_norm1_uc = net_norm1[uc_indices]
                        net_attn1_original_uc[uc_indices] = self.m.attn1(net_norm1_uc)
                        net_attn1 = (
                            style_cfg * net_attn1_original_uc
                            + (1.0 - style_cfg) * net_attn1
                        )
                    bank.clear()
                self.style_reference_states.is_written = False
            if net_attn1 is None:
                net_attn1 = self.m.attn1(net_norm1)
            net = net_attn1.to(net.dtype) + net
            net = self.m.attn2(self.m.norm2(net), context=context) + net
            net = self.m.ff(self.m.norm3(net)) + net
            return net
        raise ValueError(
            "`SpatialTransformerHooks` is not enabled, "
            "so its `forward` method should not be called."
        )

    # callbacks

    def before_unet_forward(
        self,
        net: Tensor,
        unet: nn.Module,
        all_hooks: List["SpatialTransformerHooks"],
        *,
        timesteps: Tensor,
        context: Optional[Tensor],
        is_controlnet: bool,
    ) -> None:
        if self.tome_info is not None:
            self.tome_info["size"] = net.shape[-2:]
        if self.style_reference_states is not None:
            if is_controlnet:
                return
            if self.style_reference_states.in_write:
                return
            if self.style_reference_states.is_written:
                return
            if self.style_reference_states.get_ref_net is None:
                return

            attn_ms = [hooks.m for hooks in all_hooks]
            pivots = [-m.norm1.normalized_shape[0] for m in attn_ms]
            sorted_indices: List[int] = np.argsort(pivots).tolist()
            sorted_hooks = [all_hooks[i] for i in sorted_indices]
            num_hooks = len(sorted_hooks)
            for i, hooks in enumerate(sorted_hooks):
                if hooks.style_reference_states is not None:
                    w = i / num_hooks
                    ref_w = hooks.style_reference_states.reference_weight
                    hooks.style_reference_states.enable_write = ref_w > w
                    hooks.style_reference_states.mode = ReferenceMode.WRITE
                    hooks.style_reference_states.in_write = True

            ref_net = self.style_reference_states.get_ref_net(timesteps)
            unet.forward(ref_net, timesteps=timesteps, context=context)
            for hooks in sorted_hooks:
                if hooks.style_reference_states is not None:
                    hooks.style_reference_states.mode = ReferenceMode.READ
                    hooks.style_reference_states.in_write = False


class SpatialTransformerBlock(Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        feedforward_multiplier: float = 4.0,
        feedforward_activation: str = "geglu",
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        hooks_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_split_chunk=attn_split_chunk,
        )
        latent_dim = round(query_dim * feedforward_multiplier)
        self.ff = FeedForward(
            query_dim,
            latent_dim,
            dropout,
            activation=feedforward_activation,
            add_last_dropout=False,
        )
        self.attn2 = CrossAttention(
            query_dim=query_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            attn_split_chunk=attn_split_chunk,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint
        self.hooks = SpatialTransformerHooks(self)
        self.setup_hooks(**(hooks_kwargs or {}))

    def setup_hooks(self, **hooks_kwargs: Any) -> None:
        self.hooks.setup(**hooks_kwargs)

    def forward(self, net: Tensor, context: Optional[Tensor] = None) -> Tensor:
        inputs = (net,) if context is None else (net, context)
        return gradient_checkpoint(
            self._forward,
            inputs=inputs,
            params=self.parameters(),
            enabled=self.use_checkpoint,
        )

    def _forward(self, net: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if self.hooks.enabled:
            return self.hooks.forward(net, context)
        net = self.attn1(self.norm1(net)) + net
        net = self.attn2(self.norm2(net), context=context) + net
        net = self.ff(self.norm3(net)) + net
        return net


class SpatialTransformer(Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear: bool = False,
        use_checkpoint: bool = False,
        attn_split_chunk: Optional[int] = None,
        hooks_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels, 1.0e-6, affine=True)
        self.use_linear = use_linear
        latent_channels = num_heads * head_dim
        if not use_linear:
            self.to_latent = HijackConv2d(in_channels, latent_channels, 1, 1, 0)
        else:
            self.to_latent = HijackLinear(in_channels, latent_channels)
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    latent_channels,
                    num_heads,
                    head_dim,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    attn_split_chunk=attn_split_chunk,
                    hooks_kwargs=hooks_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.from_latent = zero_module(
            HijackConv2d(latent_channels, in_channels, 1, 1, 0)
            if not use_linear
            else HijackLinear(in_channels, latent_channels)
        )

    def setup_hooks(self, **hooks_kwargs: Any) -> None:
        for block in self.blocks:
            block.setup_hooks(**hooks_kwargs)

    def forward(self, net: Tensor, context: Optional[Tensor]) -> Tensor:
        inp = net
        b, c, h, w = net.shape
        net = self.norm(net)
        if not self.use_linear:
            net = self.to_latent(net)
        net = net.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.use_linear:
            net = self.to_latent(net)
        for block in self.blocks:
            net = block(net, context=context)
        if self.use_linear:
            net = self.from_latent(net)
        net = net.permute(0, 2, 1).contiguous()
        net = net.view(b, c, h, w)
        if not self.use_linear:
            net = self.from_latent(net)
        return inp + net


class HooksCallback(Protocol):
    def __call__(
        self,
        current: SpatialTransformerHooks,
        all_hooks: List[SpatialTransformerHooks],
    ) -> None:
        pass


def walk_spatial_transformer_blocks(
    m: nn.Module,
    fn: Callable[[SpatialTransformerBlock], None],
) -> None:
    for child in m.modules():
        if isinstance(child, SpatialTransformerBlock):
            fn(child)


def walk_spatial_transformer_hooks(
    m: nn.Module,
    fn: Optional[HooksCallback] = None,
) -> List[SpatialTransformerHooks]:
    all_hooks: List[SpatialTransformerHooks] = []
    walk_spatial_transformer_blocks(m, lambda block: all_hooks.append(block.hooks))
    if fn is not None:
        for hooks in all_hooks:
            fn(hooks, all_hooks)
    return all_hooks


__all__ = [
    "MixedStackedEncoder",
    "SpatialTransformerHooks",
    "SpatialTransformerBlock",
    "SpatialTransformer",
    "walk_spatial_transformer_blocks",
    "walk_spatial_transformer_hooks",
]
