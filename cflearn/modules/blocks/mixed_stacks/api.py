import math
import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import safe_execute

from .poolers import BertPooler
from .poolers import SequencePooler
from .feedforward import FFN
from .token_mixers import TokenMixerBase
from ..norms import NormFactory
from ..common import Lambda
from ..customs import Linear
from ..customs import DropPath
from ..high_level import PreNorm
from ....misc.toolkit import interpolate


class MixingBlock(Module):
    def __init__(
        self,
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
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
            return factory.make(latent_dim, **(norm_kwargs or {}))

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
                "num_tokens": num_tokens,
                "latent_dim": latent_dim,
                "dropout": dropout,
            }
        )
        self.token_norm = _make_norm()
        token_mixing_base = TokenMixerBase.get(token_mixing_type)
        self.token_mixing = safe_execute(token_mixing_base, token_mixing_config)
        if token_mixing_dropout is None:
            token_mixing_dropout = dropout
        self.token_mixing_dropout = nn.Dropout(token_mixing_dropout)
        # channel mixing
        if channel_mixing_config is None:
            channel_mixing_config = {}
        channel_mixing_config.update(
            {
                "in_dim": latent_dim,
                "latent_dim": feedforward_dim,
                "dropout": dropout,
            }
        )
        self.residual_after_norm = residual_after_norm
        self.channel_norm = _make_norm()
        self.channel_mixing = FFN.make(channel_mixing_type, channel_mixing_config)

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if self.norm_position == "pre_norm":
            return self._pre_norm_forward(net, hw, determinate=determinate, **kwargs)
        if self.norm_position == "post_norm":
            return self._post_norm_forward(net, hw, determinate=determinate, **kwargs)
        raise ValueError(f"unrecognized norm_position '{self.norm_position}' occurred")

    def _pre_norm_forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        # token mixing
        token_mixing_kw = dict(hw=hw, determinate=determinate)
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
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        # token mixing
        token_mixing_kw = dict(hw=hw, determinate=determinate)
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
        num_history: int,
        dropout: float = 0.0,
        *,
        num_heads: int,
        is_vision: bool,
        enable: bool = True,
    ):
        super().__init__()
        self.pos_drop = None
        self.pos_encoding = None
        if enable:
            self.pos_drop = nn.Dropout(p=dropout)
            self.pos_encoding = nn.Parameter(torch.zeros(1, num_history, dim))
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self.num_heads = num_heads
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
            pos_encoding = self.interpolate_pos_encoding(net, hwp)
        else:
            pos_encoding = self.pos_encoding[:, : net.shape[1] - self.num_heads]
        pos_encoding = self.pos_drop(pos_encoding)
        return net + pos_encoding

    # this is for vision positional encodings
    def interpolate_pos_encoding(
        self,
        net: Tensor,
        hwp: Optional[Tuple[int, int, int]],
    ) -> Tensor:
        pos_encoding = self.pos_encoding
        assert pos_encoding is not None
        num_current_history = net.shape[1] - self.num_heads
        num_history = pos_encoding.shape[1] - self.num_heads
        if hwp is None:
            w = h = patch_size = None
        else:
            h, w, patch_size = hwp
        if num_current_history == num_history and w == h:
            return pos_encoding
        if w is None or h is None or patch_size is None:
            raise ValueError("`hwp` should be provided for `interpolate_pos_encoding`")
        head_encoding = None
        if self.num_heads > 0:
            head_encoding = pos_encoding[:, : self.num_heads]
            pos_encoding = pos_encoding[:, self.num_heads :]
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
        dim: int,
        num_history: int,
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
        feedforward_dim_ratio: float = 1.0,
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
            num_heads = 0
            self.head_token = None
        else:
            num_heads = 1
            if aux_heads is not None:
                num_heads += len(aux_heads)
            self.head_token = nn.Parameter(torch.zeros(1, num_heads, dim))
        self.num_heads = num_heads
        # positional encoding
        num_history += num_heads
        if is_vision_positional_encoding is None:
            if use_positional_encoding:
                raise ValueError(
                    "`is_vision_positional_encoding` should be specified when "
                    "`is_vision_positional_encoding` is set to True"
                )
            is_vision_positional_encoding = False
        self.pos_encoding = PositionalEncoding(
            dim,
            num_history,
            positional_encoding_dropout,
            num_heads=num_heads,
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
        feedforward_dim = int(round(dim * feedforward_dim_ratio))
        self.mixing_blocks = ModuleList(
            [
                MixingBlock(
                    num_history,
                    dim,
                    feedforward_dim,
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
                for drop_path in dpr_list
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
                head = SequencePooler(dim, aux_heads)
            else:
                if aux_heads is not None:
                    raise ValueError(
                        "either `head_token` or `sequence` head_pooler should be used "
                        f"when `aux_heads` ({aux_heads}) is provided"
                    )
                if head_pooler == "bert":
                    head = BertPooler(dim)
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
            self.head_norm = NormFactory(norm_type).make(dim, **(norm_kwargs or {}))
            self.head = head
        else:
            self.head_norm = None
            self.head = PreNorm(
                dim,
                module=head,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
            )
        # initializations
        if self.head_token is not None:
            nn.init.trunc_normal_(self.head_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        if isinstance(m, nn.Linear) or isinstance(m, Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def pre_process(
        self,
        net: Tensor,
        *,
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        n, t, d = net.shape
        if self.head_token is not None:
            head_tokens = self.head_token.repeat([n, 1, 1])
            net = torch.cat([head_tokens, net], dim=1)
            t += 1
        if determinate:
            net = net.view(-1, *map(int, [t, d]))
        net = self.pos_encoding(net, **kwargs)
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
        hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        determinate = kwargs.pop("determinate", False)
        net = self.pre_process(net, determinate=determinate, **kwargs)
        for block in self.mixing_blocks:
            net = block(net, hw, determinate=determinate)
        net = self.post_process(net)
        return net


__all__ = [
    "MixedStackedEncoder",
]