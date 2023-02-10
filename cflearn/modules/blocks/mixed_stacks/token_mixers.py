import math
import torch

import torch.nn as nn
import torch.nn.functional as F

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


@TokenMixerBase.register("rwkv")
class RWKVTokenMixer(TokenMixerBase):
    def __init__(
        self,
        layer_idx: int,
        num_layers: int,
        num_tokens: int,
        in_dim: int,
        latent_dim: int,
    ):
        super().__init__(in_dim, num_tokens)
        latent_dim = latent_dim or in_dim
        self.ci = latent_dim

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(in_dim, latent_dim, bias=False)
        self.value = nn.Linear(in_dim, latent_dim, bias=False)
        self.receptance = nn.Linear(in_dim, latent_dim, bias=False)
        self.output = nn.Linear(latent_dim, in_dim, bias=False)

        with torch.no_grad():
            for k, v in self.state_dict().items():
                for zero_k in ["key", "receptance", "output"]:
                    if zero_k in k:
                        nn.init.zeros_(v)
                        break
                else:
                    nn.init.orthogonal_(v)

            _0_to_1 = layer_idx / (num_layers - 1)
            _1_to_almost0 = 1.0 - (layer_idx / num_layers)

            f1_begin = 3.0
            f1_end = 1.2
            f2_begin = 0.65
            f2_end = 0.4
            decay = torch.ones(latent_dim, 1)
            first_sa_layer_id = int(num_layers > 2)
            for h in range(latent_dim):
                mul = layer_idx - first_sa_layer_id
                divider = num_layers - 1 - first_sa_layer_id
                f1 = f1_begin + mul / divider * (f1_end - f1_begin)
                f2 = f2_begin + mul / divider * (f2_end - f2_begin)
                if layer_idx == first_sa_layer_id:
                    f1 += 0.5
                elif layer_idx == num_layers - 2:
                    f2 = 0.4
                elif layer_idx == num_layers - 1:
                    f2 = 0.37
                decay[h][0] = math.pow(f2, h / (latent_dim - 1) * 7.0) * f1
            # Cl, 1
            self.info_decay = nn.Parameter(decay.log())
            # 1, T
            time_curve = torch.arange(-num_tokens + 1, 1)[None]
            self.register_buffer("time_curve", time_curve)

            x = torch.ones(1, 1, in_dim)
            for i in range(in_dim):
                x[0, 0, i] = i / in_dim
            self.time_mix_k = nn.Parameter(torch.pow(x, _1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, _1_to_almost0) + 0.3 * _0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * _1_to_almost0))

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        # B, T, C
        t = net.shape[1]
        shifted = self.time_shift(net)
        net_k = net * self.time_mix_k + shifted * (1.0 - self.time_mix_k)
        net_v = net * self.time_mix_v + shifted * (1.0 - self.time_mix_v)
        net_r = net * self.time_mix_r + shifted * (1.0 - self.time_mix_r)

        # B, Cl, T
        net_k = self.key(net_k).transpose(1, 2)
        net_v = self.value(net_v).transpose(1, 2)

        # Cl, T
        exp_w = (self.info_decay.exp() * self.time_curve).exp()
        # Cl, 1, T
        exp_w = exp_w[:, None]
        # B, Cl, T
        exp_k = net_k.exp()
        net_kv = exp_k * net_v
        # B, Cl, T
        wk = F.conv1d(F.pad(exp_k, (t - 1, 0, 0, 0)), exp_w, groups=self.ci)
        net_kv = net_kv / (wk + 1.0e-6)
        wkv = F.conv1d(F.pad(net_kv, (t - 1, 0, 0, 0)), exp_w, groups=self.ci)
        # B, T, Cl
        wkv = wkv.transpose(1, 2)
        net_r = self.receptance(net_r)
        net_r = torch.sigmoid(net_r)
        rwkv = net_r * wkv
        # B, T, C
        net = self.output(rwkv)
        return net.contiguous()


__all__ = [
    "MLPTokenMixer",
    "FourierTokenMixer",
    "AttentionTokenMixer",
    "PoolTokenMixer",
    "RWKVTokenMixer",
]
