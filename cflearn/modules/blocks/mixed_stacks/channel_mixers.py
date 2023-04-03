import torch

from torch import nn
from torch import Tensor

from .schema import ChannelMixerBase
from ..convs import DepthWiseConv2d
from ..common import Lambda
from ..hijacks import HijackCustomLinear
from ..activations import GEGLU
from ..activations import Activation


@ChannelMixerBase.register("ff")
class FeedForward(ChannelMixerBase):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        dropout: float,
        activation: str = "GELU",
        add_last_dropout: bool = True,
    ):
        super().__init__(in_dim, latent_dim, dropout)
        if activation == "geglu":
            blocks = [GEGLU(in_dim, latent_dim)]
        else:
            blocks = [
                HijackCustomLinear(in_dim, latent_dim),
                Activation.make(activation),
            ]
        blocks += [nn.Dropout(dropout), HijackCustomLinear(latent_dim, in_dim)]
        if add_last_dropout:
            blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    @property
    def need_2d(self) -> bool:
        return False

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


@ChannelMixerBase.register("mix_ff")
class MixFeedForward(ChannelMixerBase):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__(in_dim, latent_dim, dropout)
        self.net = nn.Sequential(
            HijackCustomLinear(in_dim, latent_dim),
            Lambda(lambda t: t.permute(0, 3, 1, 2), "permute -> BCHW"),
            DepthWiseConv2d(latent_dim),
            Lambda(lambda t: t.flatten(2).transpose(1, 2), "transpose -> BNC"),
            nn.GELU(),
            nn.Dropout(dropout),
            HijackCustomLinear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    @property
    def need_2d(self) -> bool:
        return True

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


@ChannelMixerBase.register("rwkv")
class RWKVChannelMixer(ChannelMixerBase):
    def __init__(
        self,
        layer_idx: int,
        num_layers: int,
        in_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__(in_dim, latent_dim, dropout)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(in_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, in_dim, bias=False)
        self.receptance = nn.Linear(in_dim, in_dim, bias=False)

        with torch.no_grad():
            for k, v in self.state_dict().items():
                for zero_k in ["value", "receptance"]:
                    if zero_k in k:
                        nn.init.zeros_(v)
                        break
                else:
                    nn.init.orthogonal_(v)

            _1_to_almost0 = 1.0 - (layer_idx / num_layers)
            x = torch.ones(1, 1, in_dim)
            for i in range(in_dim):
                x[0, 0, i] = i / in_dim
            self.time_mix_k = nn.Parameter(torch.pow(x, _1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, _1_to_almost0))

    @property
    def need_2d(self) -> bool:
        return False

    def forward(self, net: Tensor) -> Tensor:
        shifted = self.time_shift(net)
        net_k = net * self.time_mix_k + shifted * (1.0 - self.time_mix_k)
        net_r = net * self.time_mix_r + shifted * (1.0 - self.time_mix_r)
        net_k = self.key(net_k)
        net_k = torch.square(torch.relu(net_k))
        net_kv = self.value(net_k)
        return torch.sigmoid(self.receptance(net_r)) * net_kv


__all__ = [
    "FeedForward",
    "MixFeedForward",
    "RWKVChannelMixer",
]
