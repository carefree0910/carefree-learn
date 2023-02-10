from abc import abstractmethod
from torch import Tensor
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Optional
from torch.nn import Module
from cftool.misc import WithRegister


token_mixers: Dict[str, Type["TokenMixerBase"]] = {}
channel_mixers: Dict[str, Type["ChannelMixerBase"]] = {}


class TokenMixerBase(Module, WithRegister["TokenMixerBase"]):
    d = token_mixers

    def __init__(self, in_dim: int, num_tokens: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_tokens = num_tokens

    @abstractmethod
    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        pass


class ChannelMixerBase(Module, WithRegister["ChannelMixerBase"]):
    d = channel_mixers

    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

    @property
    @abstractmethod
    def need_2d(self) -> bool:
        pass

    @abstractmethod
    def forward(self, net: Tensor) -> Tensor:
        pass


__all__ = [
    "TokenMixerBase",
    "ChannelMixerBase",
]
