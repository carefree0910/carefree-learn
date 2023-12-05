import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from dataclasses import dataclass
from torch.nn import Module
from cftool.misc import DataClassBase
from cftool.types import tensor_dict_type

from ..common import PrefixModules
from ...toolkit import get_device
from ...toolkit import interpolate
from ...toolkit import eval_context


TEncoder = TypeVar("TEncoder", bound=Type["IEncoder"])
TDecoder = TypeVar("TDecoder", bound=Type["IDecoder"])

encoders = PrefixModules("encoders")
decoders = PrefixModules("decoders")


class IEncoder(Module):
    in_channels: int

    def encode(self, net: Tensor) -> Tensor:
        return self(net)


@dataclass
class DecoderInputs(DataClassBase):
    """
    A universal dataclass for decoder inputs.

    > I don't know what's a better solution to make the 'abstract' class general and
    `torch.compile` friendly at the same time.
    """

    # general
    net: Tensor
    labels: Optional[Tensor] = None
    deterministic: bool = False
    # attn
    no_head: bool = False
    apply_tanh: Optional[bool] = None


class IDecoder(Module):
    cond: Optional[Module] = None
    num_classes: Optional[int] = None
    img_size: Optional[int] = None
    latent_channels: Optional[int] = None
    latent_resolution: Optional[int] = None

    def decode(self, inputs: DecoderInputs) -> Tensor:
        return self(inputs)

    def resize(self, net: Tensor, *, deterministic: bool = False) -> Tensor:
        if self.img_size is None:
            return net
        return interpolate(net, size=self.img_size, deterministic=deterministic)

    def generate_cond(self, cond_channels: int) -> None:
        if self.num_classes is None:
            self.cond = None
        else:
            msg_fmt = "`{}` should be provided for conditional modeling"
            if self.latent_channels is None:
                raise ValueError(msg_fmt.format("latent_channels"))
            if self.latent_resolution is None:
                raise ValueError(msg_fmt.format("latent_resolution"))
            self.cond = ChannelPadding(
                self.latent_channels,
                cond_channels,
                self.latent_resolution,
                num_classes=self.num_classes,
            )

    def inject_cond(self, net: Tensor, labels: Optional[Tensor]) -> Tensor:
        if self.cond is None:
            return net
        return self.cond(net, labels)


def register_encoder(name: str, **kwargs: Any) -> Callable[[TEncoder], TEncoder]:
    return encoders.register(name, **kwargs)


def register_decoder(name: str, **kwargs: Any) -> Callable[[TDecoder], TDecoder]:
    return decoders.register(name, **kwargs)


def build_encoder(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IEncoder:
    return encoders.build(name, config=config, **kwargs)


def build_decoder(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IDecoder:
    return decoders.build(name, config=config, **kwargs)


def get_latent_resolution(encoder: IEncoder, img_size: int) -> int:
    shape = 1, encoder.in_channels, img_size, img_size
    with eval_context(encoder):
        net = encoder.encode(torch.zeros(*shape, device=get_device(encoder)))
    return net.shape[2]


class EncoderDecoder(Module):
    def __init__(
        self,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder = build_encoder(encoder, config=encoder_config)
        self.decoder = build_decoder(decoder, config=decoder_config)


class VQCodebookOut(NamedTuple):
    z_e: Tensor
    z_q: Tensor
    indices: Tensor
    z_q_g: Optional[Tensor] = None

    def to_dict(self) -> tensor_dict_type:
        return self._asdict()


class VQCodebook(nn.Module):
    def __init__(self, num_code: int, code_dimension: int):
        super().__init__()
        self.num_code = num_code
        self.code_dimension = code_dimension
        self.embedding = nn.Embedding(num_code, code_dimension)
        span = 1.0 / num_code
        self.embedding.weight.data.uniform_(-span, span)

    # z_q_g : z_q with embedding gradient
    def forward(self, z_e: Tensor, *, return_z_q_g: bool = False) -> VQCodebookOut:
        inp = z_e
        z_e = z_e.permute(0, 2, 3, 1).contiguous()

        codebook = self.embedding.weight.detach()
        with torch.no_grad():
            z_e_flattened = z_e.view(-1, self.code_dimension)
            distances = (
                torch.sum(z_e_flattened**2, dim=1, keepdim=True)
                + torch.sum(codebook**2, dim=1)
                - 2.0 * torch.einsum("bd,dn->bn", z_e_flattened, codebook.t())
            )
            indices = torch.argmin(distances, dim=1)

        z_q_flattened = codebook[indices]
        z_q = z_q_flattened.view_as(z_e)
        # VQSTE in one line of code
        z_q = z_e + (z_q - z_e).detach()

        indices = indices.view(*z_q.shape[:-1])
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if not return_z_q_g:
            return VQCodebookOut(inp, z_q, indices)

        # z_q with embedding gradient
        z_q_g_flattened = self.embedding.weight[indices]
        z_q_g = z_q_g_flattened.view_as(inp)
        return VQCodebookOut(inp, z_q, indices, z_q_g)


__all__ = [
    "IEncoder",
    "IDecoder",
    "register_encoder",
    "register_decoder",
    "build_encoder",
    "build_decoder",
    "get_latent_resolution",
    "EncoderDecoder",
    "VQCodebookOut",
    "VQCodebook",
]
