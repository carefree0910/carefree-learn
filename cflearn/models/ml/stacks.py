import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from abc import ABC
from torch import Tensor
from typing import Any
from typing import Optional

from .protocol import MERGED_KEY
from .protocol import MLCoreProtocol
from ...types import tensor_dict_type
from ...trainer import TrainerState
from ...constants import PREDICTIONS_KEY
from ...modules.blocks import _get_clones
from ...modules.blocks import Lambda
from ...modules.blocks import Linear
from ...modules.blocks import PreNorm
from ...modules.blocks import Residual


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class TokenMixerFactory(ABC):
    @staticmethod
    @abstractmethod
    def make(
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        dropout: float,
        **kwargs: Any,
    ) -> nn.Module:
        pass


class MixingBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        token_mixing_factory: TokenMixerFactory,
        *,
        dropout: float = 0.0,
        norm_type: str = "batch_norm",
        **token_mixing_kwargs: Any,
    ):
        super().__init__()
        self.token_mixing = Residual(
            PreNorm(
                latent_dim,
                module=token_mixing_factory.make(
                    num_tokens,
                    latent_dim,
                    feedforward_dim,
                    dropout,
                    **token_mixing_kwargs,
                ),
                norm_type=norm_type,
            )
        )
        self.channel_mixing = Residual(
            PreNorm(
                latent_dim,
                module=FeedForward(latent_dim, feedforward_dim, dropout),
                norm_type=norm_type,
            )
        )

    def forward(self, net: Tensor) -> Tensor:
        return self.channel_mixing(self.token_mixing(net))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        num_history: int,
        dropout: float = 0.0,
        *,
        has_head_token: bool,
        enable: bool = True,
    ):
        super().__init__()
        self.pos_drop = None
        self.pos_encoding = None
        if enable:
            self.pos_drop = nn.Dropout(p=dropout)
            self.pos_encoding = nn.Parameter(torch.zeros(1, num_history, dim))
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self.has_head_token = has_head_token

    def forward(self, net: Tensor) -> Tensor:
        if self.pos_encoding is None or self.pos_drop is None:
            return net
        pos_encoding = self.interpolate_pos_encoding(net, self.pos_encoding)
        pos_encoding = self.pos_drop(pos_encoding)
        return net + pos_encoding

    # this is for vision positional encodings
    def interpolate_pos_encoding(self, net: Tensor, pos_encoding: Tensor) -> Tensor:
        head_dim = int(self.has_head_token)
        num_current_history = net.shape[1] - head_dim
        num_history = pos_encoding.shape[1] - head_dim
        if num_current_history == num_history:
            return pos_encoding
        head_encoding = None
        if self.has_head_token:
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


class MixedStackedEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_history: int,
        token_mixing_factory: TokenMixerFactory,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: str = "batch_norm",
        feedforward_dim_ratio: float = 1.0,
        use_head_token: bool = False,
        use_positional_encoding: bool = False,
        **token_mixing_kwargs: Any,
    ):
        super().__init__()
        # head token
        if not use_head_token:
            self.head_token = None
        else:
            self.head_token = nn.Parameter(torch.zeros(1, 1, dim))
        # positional encoding
        num_history += int(use_head_token)
        self.pos_encoding = PositionalEncoding(
            dim,
            num_history,
            dropout,
            has_head_token=use_head_token,
            enable=use_positional_encoding,
        )
        # core
        feedforward_dim = int(round(dim * feedforward_dim_ratio))
        mixing_block = MixingBlock(
            num_history,
            dim,
            feedforward_dim,
            token_mixing_factory,
            dropout=dropout,
            norm_type=norm_type,
            **token_mixing_kwargs,
        )
        self.mixing_blocks = _get_clones(mixing_block, num_layers)
        # initializations
        if self.head_token is not None:
            nn.init.trunc_normal_(self.head_token, std=0.02)
        self.apply(self._init_weights)
        # head
        if self.head_token is not None:
            head = Lambda(lambda x: x[:, 0], name="head_token")
        else:
            head = Lambda(lambda x: x.mean(1), name="global_average")
        self.head = PreNorm(dim, module=head, norm_type=norm_type)

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
        net = self.pos_encoding(net)
        for block in self.mixing_blocks:
            net = block(net)
        net = self.head(net)
        return net


class MixedStackedModel(MLCoreProtocol):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_history: int,
        latent_dim: int,
        token_mixing_factory: TokenMixerFactory,
        *,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: str = "batch_norm",
        feedforward_dim_ratio: float = 1.0,
        use_head_token: bool = False,
        use_positional_encoding: bool = False,
        **token_mixing_kwargs: Any,
    ):
        super().__init__(in_dim, out_dim, num_history)
        self.to_encoder = Linear(in_dim, latent_dim)
        self.encoder = MixedStackedEncoder(
            latent_dim,
            num_history,
            token_mixing_factory,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            use_head_token=use_head_token,
            use_positional_encoding=use_positional_encoding,
            **token_mixing_kwargs,
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
        net = self.to_encoder(net)
        net = self.encoder(net)
        net = self.head(net)
        return {PREDICTIONS_KEY: net}


__all__ = [
    "MixedStackedEncoder",
    "MixedStackedModel",
]
