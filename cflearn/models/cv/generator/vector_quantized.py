import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import NamedTuple
from cftool.types import tensor_dict_type

from ..decoder import make_decoder
from ..encoder import run_encoder
from ..encoder import make_encoder
from ..encoder import EncoderMixin
from ....schema import TrainerState
from ....schema import IDLModel
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import Conv2d
from ....modules.blocks import ChannelPadding


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


@IDLModel.register("vq_generator")
class VQGenerator(IDLModel):
    def __init__(
        self,
        img_size: int,
        num_code: int,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        *,
        encoder: str,
        decoder: str,
        code_dimension: int = 256,
        latent_channels: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        latent_padding_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_code = num_code
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.code_dimension = code_dimension
        self.latent_channels = latent_channels
        if encoder_config is None:
            encoder_config = {}
        if decoder_config is None:
            decoder_config = {}
        # encoder
        encoder_config.setdefault("img_size", img_size)
        encoder_config.setdefault("in_channels", in_channels)
        encoder_config.setdefault("latent_channels", latent_channels)
        self.encoder = make_encoder(encoder, encoder_config)
        assert isinstance(self.encoder, EncoderMixin)
        latent_resolution = self.encoder.latent_resolution(img_size)
        self.latent_resolution = latent_resolution
        # codebook
        self.codebook = VQCodebook(num_code, code_dimension)
        # decoder
        decoder_config.setdefault("img_size", img_size)
        decoder_config.setdefault("out_channels", out_channels or in_channels)
        decoder_config.setdefault("latent_resolution", latent_resolution)
        decoder_config.setdefault("latent_channels", latent_channels)
        decoder_config.setdefault("num_classes", num_classes)
        self.decoder = make_decoder(decoder, decoder_config)
        # latent padding
        if latent_padding_channels is None:
            self.latent_padding = None
        else:
            self.latent_padding = ChannelPadding(
                latent_channels,
                latent_padding_channels,
                latent_resolution,
            )

    def to_codebook(self, latent: Tensor) -> Tensor:
        return latent

    def from_codebook(self, z_q: Tensor) -> Tensor:
        return z_q

    def encode(self, net: Tensor) -> Tensor:
        net = self.encoder.encode({INPUT_KEY: net})
        net = self.to_codebook(net)
        net = self.codebook(net).z_q
        return net

    def decode(
        self,
        z_q: Tensor,
        *,
        labels: Optional[Tensor] = None,
        resize: bool = True,
    ) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z_q)], device=z_q.device)
        if self.latent_padding is not None:
            z_q = self.latent_padding(z_q)
        net = self.from_codebook(z_q)
        batch = {INPUT_KEY: net, LABEL_KEY: labels}
        net = self.decoder.decode(batch, resize=resize)
        return net

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        return_z_q_g: bool = False,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z_e = run_encoder(self.encoder, batch_idx, batch, state, **kwargs)[LATENT_KEY]
        net = self.to_codebook(z_e)
        out = self.codebook(net, return_z_q_g=return_z_q_g)
        z_q = self.from_codebook(out.z_q)
        out = out._replace(z_q=z_q)
        net = self.decode(z_q, labels=batch.get(LABEL_KEY))
        results = {PREDICTIONS_KEY: net}
        results.update(out.to_dict())
        return results

    def get_code_indices(self, net: Tensor, **kwargs: Any) -> Tensor:
        z_e = self.encoder.encode({INPUT_KEY: net}, **kwargs)
        indices = self.codebook(z_e).indices
        return indices

    def get_code(self, code_indices: Tensor) -> Tensor:
        code_indices = code_indices.squeeze(1)
        z_q = self.codebook.embedding(code_indices.to(self.device))
        return z_q.permute(0, 3, 1, 2)

    def reconstruct_from(
        self,
        code_indices: Tensor,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
        use_one_hot: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        z_q = self.get_code(code_indices)
        if use_one_hot:
            one_hot = torch.zeros_like(z_q)
            i = int(round(0.5 * z_q.shape[2]))
            j = int(round(0.5 * z_q.shape[3]))
            one_hot[..., i, j] = z_q[..., i, j]
            z_q = one_hot
        if labels is None and class_idx is not None:
            labels = torch.full([len(z_q)], class_idx, device=self.device)
        return self.decode(z_q, labels=labels, **kwargs)

    def sample_codebook(
        self,
        *,
        code_indices: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
        class_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        if code_indices is None:
            if num_samples is None:
                raise ValueError("either `indices` or `num_samples` should be provided")
            code_indices = torch.randint(self.num_code, [num_samples])
        code_indices = code_indices.view(-1, 1, 1, 1)
        resolution = self.latent_resolution
        tiled = code_indices.repeat([1, 1, resolution, resolution])
        if class_idx is not None:
            kwargs["labels"] = torch.full([len(code_indices)], class_idx)
        kwargs.setdefault("use_one_hot", True)
        net = self.reconstruct_from(tiled, **kwargs)
        return net, code_indices


@VQGenerator.register("vqgan_generator")
class VQGANGenerator(VQGenerator):
    def __init__(
        self,
        img_size: int,
        num_code: int = 16384,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        *,
        code_dimension: int = 256,
        latent_channels: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        latent_padding_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__(
            img_size,
            num_code,
            in_channels,
            out_channels,
            encoder="vqgan",
            decoder="vqgan",
            code_dimension=code_dimension,
            latent_channels=latent_channels,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            latent_padding_channels=latent_padding_channels,
            num_classes=num_classes,
        )
        self.q_conv = Conv2d(latent_channels, code_dimension, kernel_size=1)
        self.post_q_conv = Conv2d(code_dimension, latent_channels, kernel_size=1)

    def to_codebook(self, latent: Tensor) -> Tensor:
        return self.q_conv(latent)

    def from_codebook(self, z_q: Tensor) -> Tensor:
        return self.post_q_conv(z_q)


__all__ = [
    "VQGenerator",
    "VQGANGenerator",
]
