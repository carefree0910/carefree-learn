import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Optional
from torch.fft import fft
from cftool.misc import safe_execute

from .schema import TokenMixerBase
from .channel_mixers import FeedForward
from ..common import Lambda
from ..attentions import Attention


@TokenMixerBase.register("mlp")
class MLPTokenMixer(TokenMixerBase):
    def __init__(self, in_dim: int, num_tokens: int, *, dropout: float = 0.1):
        super().__init__(in_dim, num_tokens)
        self.net = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2), name="to_token_mixing"),
            FeedForward(num_tokens, num_tokens, dropout),
            Lambda(lambda x: x.transpose(1, 2), name="to_channel_mixing"),
        )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.net(net)


@TokenMixerBase.register("fourier")
class FourierTokenMixer(TokenMixerBase):
    def __init__(self, in_dim: int, num_tokens: int):
        super().__init__(in_dim, num_tokens)
        self.net = Lambda(lambda x: fft(fft(x, dim=-1), dim=-2).real, name="fourier")

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.net(net)


@TokenMixerBase.register("attention")
class AttentionTokenMixer(TokenMixerBase):
    def __init__(
        self,
        in_dim: int,
        num_tokens: int,
        *,
        attention_type: str = "basic",
        **attention_kwargs: Any,
    ):
        super().__init__(in_dim, num_tokens)
        attention_kwargs.setdefault("bias", False)
        attention_kwargs.setdefault("num_heads", 8)
        attention_kwargs["input_dim"] = in_dim
        attention_kwargs.setdefault("is_self_attention", True)
        self.net = safe_execute(Attention.get(attention_type), attention_kwargs)

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        kw = dict(hw=hw, mask=mask, determinate=determinate)
        return self.net(net, net, net, **kw).output


@TokenMixerBase.register("pool")
class PoolTokenMixer(TokenMixerBase):
    def __init__(self, in_dim: int, num_tokens: int, *, pool_size: int = 3):
        super().__init__(in_dim, num_tokens)
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.pool(net) - net


__all__ = [
    "MLPTokenMixer",
    "FourierTokenMixer",
    "AttentionTokenMixer",
    "PoolTokenMixer",
]
