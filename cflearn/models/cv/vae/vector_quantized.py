import torch

import torch.nn as nn

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
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ....modules.blocks import ChannelPadding


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

    def forward(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_q, indices = VQSTE.apply(z_e, self.embedding.weight.detach())
        indices = indices.view(*z_q.shape[:-1])
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, indices


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
        latent_padding_channels: Optional[int] = 16,
        num_classes: Optional[int] = None,
        encoder_configs: Optional[Dict[str, Any]] = None,
        decoder_configs: Optional[Dict[str, Any]] = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        # encoder
        if encoder_configs is None:
            encoder_configs = {}
        encoder_configs["img_size"] = img_size
        encoder_configs["in_channels"] = in_channels
        encoder_configs["num_downsample"] = num_downsample
        self.encoder = EncoderBase.make(encoder, config=encoder_configs)
        # latent
        self.num_code = num_code
        self.map_dim = f_map_dim(img_size, num_downsample)
        self.latent_channels = self.encoder.latent_channels
        self.codebook = VQCodebook(num_code, self.latent_channels)
        decoder_in_channels = self.latent_channels
        if latent_padding_channels is None:
            self.latent_padding = None
        else:
            decoder_in_channels += latent_padding_channels
            self.latent_padding = ChannelPadding(latent_padding_channels, self.map_dim)
        # decoder
        if decoder_configs is None:
            decoder_configs = {}
        decoder_configs["img_size"] = img_size
        decoder_configs["latent_channels"] = decoder_in_channels
        decoder_configs["latent_resolution"] = self.map_dim
        decoder_configs["num_upsample"] = num_downsample
        decoder_configs["out_channels"] = out_channels or in_channels
        decoder_configs["num_classes"] = num_classes
        self.decoder = DecoderBase.make(decoder, config=decoder_configs)

    def _decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        if labels is None and self.num_classes is not None:
            labels = torch.randint(self.num_classes, [len(z)], device=z.device)
        if self.latent_padding is not None:
            z = self.latent_padding(z)
        batch = {INPUT_KEY: z, LABEL_KEY: labels}
        net = self.decoder.decode(batch, **kwargs)[PREDICTIONS_KEY]
        return torch.tanh(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z_e = self.encoder.encode(batch, **kwargs)[LATENT_KEY]
        z_q, indices = self.codebook(z_e)
        z_q_g_flatten = self.codebook.embedding.weight[indices]
        shape = -1, *z_q.shape[2:], self.latent_channels
        z_q_g = z_q_g_flatten.view(*shape).permute(0, 3, 1, 2).contiguous()
        kwargs[LABEL_KEY] = batch[LABEL_KEY]
        net = self._decode(z_q, **kwargs)
        return {PREDICTIONS_KEY: net, "z_e": z_e, "z_q_g": z_q_g, "indices": indices}

    def get_code_indices(self, net: Tensor, **kwargs: Any) -> Tensor:
        batch = {INPUT_KEY: net}
        z_e = self.encoder.encode(batch, **kwargs)[LATENT_KEY]
        _, indices = self.codebook(z_e)
        return indices

    def reconstruct_from(
        self,
        code_indices: Tensor,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
        use_one_hot: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        z_q = self.codebook.embedding(code_indices.to(self.device))
        z_q = z_q.permute(0, 3, 1, 2)
        if use_one_hot:
            one_hot = torch.zeros_like(z_q)
            i = int(round(0.5 * z_q.shape[2]))
            j = int(round(0.5 * z_q.shape[3]))
            one_hot[..., i, j] = z_q[..., i, j]
            z_q = one_hot
        if labels is None and class_idx is not None:
            labels = torch.full([len(z_q)], class_idx, device=self.device)
        return self._decode(z_q, labels=labels, **kwargs)

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
        code_indices = code_indices.view(-1, 1, 1)
        tiled = code_indices.repeat([1, self.map_dim, self.map_dim])
        if class_idx is not None:
            kwargs["labels"] = torch.full([num_samples], class_idx)
        kwargs.setdefault("use_one_hot", True)
        net = self.reconstruct_from(tiled, **kwargs)
        return net, code_indices


__all__ = ["VQVAE"]
