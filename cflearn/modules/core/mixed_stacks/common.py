from abc import abstractmethod
from torch import Tensor
from typing import Any
from typing import Type
from typing import Tuple
from typing import Callable
from typing import Optional
from torch.nn import Module

from ...common import PrefixModules


TTMixer = Type["ITokenMixer"]
TCMixer = Type["IChannelMixer"]

token_mixers = PrefixModules("token_mixer")
channel_mixers = PrefixModules("channel_mixer")


class ITokenMixer(Module):
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


class IChannelMixer(Module):
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


def register_token_mixer(name: str, **kwargs: Any) -> Callable[[TTMixer], TTMixer]:
    return token_mixers.register(name, **kwargs)


def register_channel_mixer(name: str, **kwargs: Any) -> Callable[[TCMixer], TCMixer]:
    return channel_mixers.register(name, **kwargs)


def build_token_mixer(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> ITokenMixer:
    return token_mixers.build(name, config=config, **kwargs)


def build_channel_mixer(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IChannelMixer:
    return channel_mixers.build(name, config=config, **kwargs)


__all__ = [
    "ITokenMixer",
    "IChannelMixer",
    "register_token_mixer",
    "register_channel_mixer",
    "build_token_mixer",
    "build_channel_mixer",
]
