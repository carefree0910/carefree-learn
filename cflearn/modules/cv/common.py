import torch
import random

from abc import abstractmethod
from abc import ABCMeta
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

from ..core import Conv2d
from ..core import ChannelPadding
from ..common import PrefixModules
from ...toolkit import slerp
from ...toolkit import get_device
from ...toolkit import interpolate
from ...toolkit import eval_context


TEncoder = TypeVar("TEncoder", bound=Type["IEncoder"])
TDecoder = TypeVar("TDecoder", bound=Type["IDecoder"])
TGenerator = TypeVar("TGenerator", bound=Type["IGenerator"])
TDiscriminator = TypeVar("TDiscriminator", bound=Type["IDiscriminator"])
TAutoRegressor = TypeVar("TAutoRegressor", bound=Type["IAutoRegressor"])

encoders = PrefixModules("encoders")
decoders = PrefixModules("decoders")
generators = PrefixModules("generators")
discriminators = PrefixModules("discriminators")
auto_regressors = PrefixModules("auto_regressors")


class IEncoder(Module):
    """
    An `IEncoder` is a module that takes an image tensor as input and outputs a latent.
    """

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
    z: Tensor
    labels: Optional[Tensor] = None
    deterministic: bool = False
    apply_tanh: Optional[bool] = None
    # attn
    no_head: bool = False
    # vq
    apply_codebook: bool = True


class IConditional(Module):
    num_classes: Optional[int] = None

    @property
    def is_conditional(self) -> bool:
        return self.num_classes is not None

    def get_sample_labels(
        self,
        num_samples: int,
        class_idx: Optional[int] = None,
    ) -> Optional[Tensor]:
        if self.num_classes is None:
            return None
        if class_idx is not None:
            return torch.full([num_samples], class_idx, device=get_device(self))
        return torch.randint(self.num_classes, [num_samples], device=get_device(self))


class IDecoder(IConditional):
    """
    An `IDecoder` is a module that takes a latent tensor as input and outputs an image.

    The latent tensor can be either a 1d tensor or a 2d tensor.
    """

    cond: Optional[Module]
    img_size: Optional[int] = None
    latent_channels: Optional[int] = None
    latent_resolution: Optional[int] = None
    apply_tanh: bool = False

    def decode(self, inputs: DecoderInputs) -> Tensor:
        net = self(inputs)
        apply_tanh = inputs.apply_tanh
        if apply_tanh or (apply_tanh is None and self.apply_tanh):
            net = torch.tanh(net)
        return net

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
        cond = getattr(self, "cond", None)
        if cond is None:
            return net
        return cond(net, labels)


class IGenerator(IConditional, metaclass=ABCMeta):
    """
    An `IGenerator` is a module that can generate images.

    It differs from `IDecoder` in that it sometimes can reconstruct images as well. Either
    way, it still often uses `IDecoder` as its generate model.
    """

    latent_dim: int

    @abstractmethod
    def generate_z(self, num_samples: int) -> Tensor:
        pass

    def decode(self, inputs: DecoderInputs) -> Tensor:
        # since `IGenerator` often uses `IDecoder` as its generate model,
        # we don't need to worry about post-processing here
        return self(inputs)

    def sample(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        z = self.generate_z(num_samples)
        if labels is None:
            labels = self.get_sample_labels(num_samples, class_idx)
        elif class_idx is not None:
            msg = "`class_idx` should not be provided when `labels` is provided"
            raise ValueError(msg)
        return self.decode(DecoderInputs(z=z, labels=labels))

    def reconstruct(
        self,
        net: Tensor,
        *,
        labels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        return None

    def interpolate(
        self,
        num_samples: int,
        *,
        class_idx: Optional[int] = None,
        use_slerp: bool = False,
    ) -> Tensor:
        z1 = self.generate_z(num_samples)
        z2 = self.generate_z(num_samples)
        shape = z1.shape
        z1 = z1.view(num_samples, -1)
        z2 = z2.view(num_samples, -1)
        ratio = torch.linspace(0.0, 1.0, num_samples, device=get_device(self))[:, None]
        z = slerp(z1, z2, ratio) if use_slerp else ratio * z1 + (1.0 - ratio) * z2
        z = z.view(num_samples, *shape[1:])
        if class_idx is None and self.num_classes is not None:
            class_idx = random.randint(0, self.num_classes - 1)
        labels = self.get_sample_labels(num_samples, class_idx)
        return self.decode(DecoderInputs(z=z, labels=labels))


class IGaussianGenerator(IGenerator):
    """
    An `IGaussianGenerator` futher specifies that the latent space is a (standard)
    Gaussian distribution.
    """

    def generate_z(self, num_samples: int) -> Tensor:
        return torch.randn(num_samples, self.latent_dim, device=get_device(self))


class DiscriminatorOutput(NamedTuple):
    output: Tensor
    cond_logits: Optional[Tensor] = None


class IDiscriminator(Module):
    """
    An `IDiscriminator` is typically used in GANs.
    """

    num_classes: Optional[int]

    def generate_cond(self, out_channels: int) -> None:
        if self.num_classes is None:
            self.cond = None
        else:
            kw = dict(kernel_size=4, padding=1, stride=1)
            self.cond = Conv2d(out_channels, self.num_classes, **kw)  # type: ignore


class IAutoRegressor(IConditional):
    """
    An `IAutoRegressor` is a module that can generate images in an auto-regressive way.

    It is often used in VQ-VAEs.
    """

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        *,
        img_size: int,
        labels: Optional[Tensor] = None,
        class_idx: Optional[int] = None,
    ) -> Tensor:
        pass


def register_encoder(name: str, **kwargs: Any) -> Callable[[TEncoder], TEncoder]:
    return encoders.register(name, **kwargs)


def register_decoder(name: str, **kwargs: Any) -> Callable[[TDecoder], TDecoder]:
    return decoders.register(name, **kwargs)


def register_generator(name: str, **kwargs: Any) -> Callable[[TGenerator], TGenerator]:
    return generators.register(name, **kwargs)


def register_discriminator(
    name: str,
    **kwargs: Any,
) -> Callable[[TDiscriminator], TDiscriminator]:
    return discriminators.register(name, **kwargs)


def register_auto_regressor(
    name: str,
    **kwargs: Any,
) -> Callable[[TAutoRegressor], TAutoRegressor]:
    return auto_regressors.register(name, **kwargs)


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


def build_generator(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IGaussianGenerator:
    return generators.build(name, config=config, **kwargs)


def build_discriminator(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IDiscriminator:
    return discriminators.build(name, config=config, **kwargs)


def build_auto_regressor(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> IAutoRegressor:
    return auto_regressors.build(name, config=config, **kwargs)


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
    "encoders",
    "decoders",
    "generators",
    "discriminators",
    "IEncoder",
    "DecoderInputs",
    "IDecoder",
    "IGenerator",
    "IGaussianGenerator",
    "DiscriminatorOutput",
    "IDiscriminator",
    "register_encoder",
    "register_decoder",
    "register_generator",
    "register_discriminator",
    "build_encoder",
    "build_decoder",
    "build_generator",
    "build_discriminator",
    "get_latent_resolution",
    "EncoderDecoder",
    "VQCodebookOut",
    "VQCodebook",
]
