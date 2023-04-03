import torch


from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from typing import NamedTuple
from cftool.types import tensor_dict_type

from .encoder import make_encoder
from .encoder import IEncoder
from .encoder import EncoderMixin
from .decoder import make_decoder
from .decoder import IDecoder
from ...misc.toolkit import auto_num_layers


class EncoderDecoderMixin:
    encoder: IEncoder
    decoder: IDecoder

    def resize(self, net: Tensor) -> Tensor:
        return self.decoder.resize(net)

    def encode(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        return self.encoder.encode(batch, **kwargs)

    def decode(
        self,
        batch: tensor_dict_type,
        *,
        resize: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        net = self.decoder.decode(batch, **kwargs)
        if resize:
            net = self.resize(net)
        return net


class EncoderDecoder(nn.Module, EncoderDecoderMixin):
    def __init__(
        self,
        is_1d: bool,
        in_channels: int,
        out_channels: Optional[int] = None,
        target_downsample: Optional[int] = None,
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        *,
        latent: int = 128,
        decoder_latent: Optional[int] = None,
        img_size: Optional[int] = None,
        min_size: int = 2,
        num_downsample: Optional[int] = None,
        num_upsample: Optional[int] = None,
        latent_resolution: Optional[int] = None,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # up / down sample stuffs
        if num_downsample is None:
            if img_size is None:
                raise ValueError(
                    "either `img_size` or `num_downsample` should be provided "
                    "(when `num_downsample` is not provided, it will be inferred "
                    "automatically with `img_size` & `min_size`)"
                )
            args = img_size, min_size, target_downsample
            num_downsample = auto_num_layers(*args, use_stride=encoder == "vanilla")
        if latent_resolution is None and img_size is None:
            raise ValueError(
                "either `img_size` or `latent_resolution` should be provided "
                "in `VanillaVAEBase` (when `latent_resolution` is not provided, "
                "it will be inferred automatically with `img_size`)"
            )
        # properties
        self.is_1d = is_1d
        self.latent = latent
        self.decoder_latent = decoder_latent or latent
        self.latent_key = "latent_dim" if is_1d else "latent_channels"
        self.img_size = img_size
        self.latent_padding_channels = latent_padding_channels
        self.num_classes = num_classes
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample
        # encoder
        if encoder_config is None:
            encoder_config = {}
        encoder_config.setdefault("img_size", img_size)
        encoder_config.setdefault("in_channels", in_channels)
        encoder_config.setdefault("num_downsample", num_downsample)
        encoder_config[self.latent_key] = latent
        self.encoder = make_encoder(encoder, encoder_config, is_1d=is_1d)  # type: ignore
        if isinstance(self.encoder, EncoderMixin) and img_size is not None:
            latent_resolution = self.encoder.latent_resolution(img_size)
        # decoder
        if decoder_config is None:
            decoder_config = {}
        decoder_config.setdefault("img_size", img_size)
        decoder_config.setdefault("num_classes", num_classes)
        decoder_config.setdefault("out_channels", out_channels or in_channels)
        decoder_config.setdefault("num_upsample", num_upsample)
        decoder_config.setdefault("latent_resolution", latent_resolution)
        decoder_config[self.latent_key] = self.decoder_latent
        self.decoder = make_decoder(decoder, decoder_config, is_1d=is_1d)
        self.latent_resolution = self.decoder.latent_resolution


class PureEncoderDecoder(nn.Module, EncoderDecoderMixin):
    def __init__(
        self,
        *,
        is_1d: bool,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder = make_encoder(encoder, encoder_config or {}, is_1d=is_1d)
        self.decoder = make_decoder(decoder, decoder_config or {}, is_1d=is_1d)


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
    "EncoderDecoder",
    "PureEncoderDecoder",
    "VQCodebook",
]
