import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from PIL import ImageDraw
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from torch.autograd import Function

from ..encoder import EncoderBase
from ..decoder import DecoderBase
from ..toolkit import f_map_dim
from ..toolkit import auto_num_layers
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import to_torch


class VQSTE(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        net, codebook = args

        with torch.no_grad():
            net_shape = net.shape
            code_dimension = codebook.shape[1]
            diff = net.view(-1, 1, code_dimension) - codebook[None, ...]
            distances = (diff ** 2).sum(2)
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*net_shape[:-1])
            ctx.mark_non_differentiable(indices)
            ctx.mark_non_differentiable(indices_flatten)
            ctx.save_for_backward(indices_flatten, codebook)

        codes_flatten = codebook[indices_flatten]
        codes = codes_flatten.view_as(net)
        return codes, indices_flatten

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        grad_net, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_net = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            code_dimension = codebook.shape[1]
            grad_output_flatten = grad_output.contiguous().view(-1, code_dimension)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return grad_net, grad_codebook


class VQCodebook(nn.Module):
    def __init__(self, num_code: int, code_dimension: int):
        super().__init__()
        self.embedding = nn.Embedding(num_code, code_dimension)
        span = 1.0 / num_code
        self.embedding.weight.data.uniform_(-span, span)

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_q, indices = VQSTE.apply(z_e, self.embedding.weight.detach())
        indices = indices.view(*z_q.shape[:-1])
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q_g_flatten = self.embedding.weight[indices]
        z_q_g = z_q_g_flatten.view_as(z_e).permute(0, 3, 1, 2).contiguous()
        return z_q, z_q_g, indices


@ModelProtocol.register("vq_vae")
class VQVAE(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        num_code: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        min_size: int = 4,
        target_downsample: int = 4,
        encoder_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
    ):
        super().__init__()
        self.img_size = img_size
        num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        # encoder
        if encoder_configs is None:
            encoder_configs = {}
        encoder_configs["img_size"] = img_size
        encoder_configs["in_channels"] = in_channels
        encoder_configs["num_downsample"] = num_downsample
        self.encoder = EncoderBase.make(encoder, **encoder_configs)
        # latent
        self.num_code = num_code
        self.f_map_dim = f_map_dim(img_size, num_downsample)
        self.latent_channels = self.encoder.latent_channels
        self.codebook = VQCodebook(num_code, self.latent_channels)
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = self.latent_channels
        decoder_configs["latent_resolution"] = self.f_map_dim
        decoder_configs["num_upsample"] = num_downsample
        decoder_configs["out_channels"] = out_channels or in_channels
        self.decoder = DecoderBase.make(decoder, **decoder_configs)

    def _decode(self, z: Tensor, **kwargs: Any) -> Tensor:
        decoded = self.decoder.decode({INPUT_KEY: z}, **kwargs)[PREDICTIONS_KEY]
        net = F.interpolate(decoded, size=self.img_size)
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z_e = self.encoder.encode(batch, **kwargs)[PREDICTIONS_KEY]
        z_q, z_q_g, indices = self.codebook(z_e)
        net = self._decode(z_q, **kwargs)
        return {PREDICTIONS_KEY: net, "z_e": z_e, "z_q_g": z_q_g, "indices": indices}

    def sample_codebook(
        self,
        *,
        indices: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        if indices is None:
            if num_samples is None:
                raise ValueError("either `indices` or `num_samples` should be provided")
            indices = torch.randint(self.num_code, [num_samples])
        indices = indices.view(-1, 1, 1, 1)
        tiled = torch.tile(indices, [1, self.f_map_dim, self.f_map_dim]).view(-1)
        z_q = self.codebook.embedding(tiled.to(self.device))
        z_q = z_q.view(-1, self.latent_channels, self.f_map_dim, self.f_map_dim)
        net = self._decode(z_q, **kwargs)
        return net, indices

    @staticmethod
    def make_map_from(indices: Tensor) -> Tensor:
        images = []
        for idx in indices.view(-1).tolist():
            img = Image.new("RGB", (28, 28), (250, 250, 250))
            draw = ImageDraw.Draw(img)
            draw.text((12, 9), str(idx), (0, 0, 0))
            images.append(to_torch(np.array(img).transpose([2, 0, 1])))
        return torch.stack(images).float()


__all__ = ["VQVAE"]
