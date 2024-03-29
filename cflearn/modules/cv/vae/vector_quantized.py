import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from ..common import build_encoder
from ..common import build_decoder
from ..common import get_latent_resolution
from ..common import VQCodebook
from ..common import IConditional
from ..common import DecoderInputs
from ...core import ChannelPadding
from ...common import register_module
from ....toolkit import get_device
from ....toolkit import auto_num_layers
from ....constants import PREDICTIONS_KEY


@register_module("vq_vae")
class VQVAE(IConditional):
    def __init__(
        self,
        img_size: int,
        num_codes: int,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: Optional[int] = None,
        min_size: int = 8,
        target_downsample: Optional[int] = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        code_dimension: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        latent_padding_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        apply_tanh: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_codes = num_codes
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_classes = num_classes
        self.code_dimension = code_dimension
        if num_downsample is None:
            args = img_size, min_size, target_downsample
            num_downsample = auto_num_layers(*args, use_stride=encoder == "vanilla")
        # encoder
        if encoder_config is None:
            encoder_config = {}
        encoder_config["num_downsample"] = num_downsample
        encoder_config.setdefault("img_size", img_size)
        encoder_config.setdefault("in_channels", in_channels)
        encoder_config.setdefault("latent_channels", code_dimension)
        self.encoder = build_encoder(encoder, config=encoder_config)
        latent_resolution = get_latent_resolution(self.encoder, img_size)
        self.latent_resolution = latent_resolution
        # codebook
        self.codebook = VQCodebook(num_codes, code_dimension)
        # decoder
        if decoder_config is None:
            decoder_config = {}
        decoder_config["apply_tanh"] = apply_tanh
        decoder_config["num_upsample"] = num_downsample
        decoder_config.setdefault("img_size", img_size)
        decoder_config.setdefault("out_channels", out_channels or in_channels)
        decoder_config.setdefault("latent_resolution", latent_resolution)
        decoder_config.setdefault("latent_channels", code_dimension)
        decoder_config.setdefault("num_classes", num_classes)
        self.decoder = build_decoder(decoder, config=decoder_config)
        # latent padding
        if latent_padding_channels is None:
            self.latent_padding = None
        else:
            self.latent_padding = ChannelPadding(
                code_dimension,
                latent_padding_channels,
                latent_resolution,
            )

    def to_codebook(self, latent: Tensor) -> Tensor:
        return latent

    def from_codebook(self, z_q: Tensor) -> Tensor:
        return z_q

    def encode(self, net: Tensor) -> Tensor:
        net = self.encoder.encode(net)
        net = self.to_codebook(net)
        net = self.codebook(net).z_q
        return net

    def decode(
        self,
        z_q: Tensor,
        *,
        labels: Optional[Tensor] = None,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z_q)], device=z_q.device)
        if self.latent_padding is not None:
            z_q = self.latent_padding(z_q)
        inputs = DecoderInputs(z=z_q, labels=labels, apply_tanh=apply_tanh)
        net = self.decoder.decode(inputs)
        return net

    def forward(
        self,
        net: Tensor,
        labels: Optional[Tensor] = None,
        *,
        return_z_q_g: bool = True,
    ) -> tensor_dict_type:
        z_e = self.encoder.encode(net)
        net = self.to_codebook(z_e)
        out = self.codebook(net, return_z_q_g=return_z_q_g)
        z_q = self.from_codebook(out.z_q)
        out = out._replace(z_q=z_q)
        net = self.decode(z_q, labels=labels)
        results = {PREDICTIONS_KEY: net}
        results.update(out.to_dict())
        return results

    def get_code_indices(self, net: Tensor, **kwargs: Any) -> Tensor:
        z_e = self.encoder.encode(net, **kwargs)
        net = self.to_codebook(z_e)
        indices = self.codebook(net).indices
        return indices

    def get_code(self, code_indices: Tensor) -> Tensor:
        code_indices = code_indices.squeeze(1)
        z_q = self.codebook.embedding(code_indices.to(get_device(self)))
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = self.from_codebook(z_q)
        return z_q

    def reconstruct_from(
        self,
        code_indices: Tensor,
        *,
        labels: Optional[Tensor] = None,
        class_idx: Optional[int] = None,
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
        if labels is None:
            labels = self.get_sample_labels(len(z_q), class_idx)
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
            code_indices = torch.randint(self.num_codes, [num_samples])
        code_indices = code_indices.view(-1, 1, 1, 1)
        resolution = self.latent_resolution
        tiled = code_indices.repeat([1, 1, resolution, resolution])
        if class_idx is not None:
            kwargs["labels"] = self.get_sample_labels(len(code_indices), class_idx)
        kwargs.setdefault("use_one_hot", True)
        net = self.reconstruct_from(tiled, **kwargs)
        return net, code_indices


__all__ = [
    "VQVAE",
]
