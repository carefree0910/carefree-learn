import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from ....protocol import TrainerState
from ....protocol import WithDeviceMixin
from ....constants import PREDICTIONS_KEY
from ..generator.vector_quantized import VQGenerator
from ....misc.toolkit import auto_num_layers
from ....misc.internal_.register import register_module


@register_module("vq_vae")
class VQVAE(nn.Module, WithDeviceMixin):
    def __init__(
        self,
        img_size: int,
        num_code: int,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: Optional[int] = None,
        min_size: int = 8,
        target_downsample: int = None,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        code_dimension: int = 256,
        latent_channels: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        latent_padding_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        apply_tanh: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.apply_tanh = apply_tanh
        if num_downsample is None:
            args = img_size, min_size, target_downsample
            num_downsample = auto_num_layers(*args, use_stride=encoder == "vanilla")
        # encoder
        if encoder_config is None:
            encoder_config = {}
        encoder_config["num_downsample"] = num_downsample
        # decoder
        if decoder_config is None:
            decoder_config = {}
        decoder_config["num_upsample"] = num_downsample
        # vq generator
        self.generator = VQGenerator(
            img_size,
            num_code,
            in_channels,
            out_channels,
            encoder=encoder,
            decoder=decoder,
            code_dimension=code_dimension,
            latent_channels=latent_channels,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            latent_padding_channels=latent_padding_channels,
            num_classes=num_classes,
        )
        self.latent_resolution = self.generator.latent_resolution

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        apply_tanh: Optional[bool] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = self.generator(batch_idx, batch, state, return_z_q_g=True, **kwargs)
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        if apply_tanh:
            results[PREDICTIONS_KEY] = torch.tanh(results[PREDICTIONS_KEY])
        return results

    def decode(
        self,
        z_q: Tensor,
        *,
        labels: Optional[Tensor] = None,
        apply_tanh: Optional[bool] = None,
        resize: bool = True,
    ) -> Tensor:
        net = self.generator.decode(z_q, labels=labels, resize=resize)
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        if apply_tanh:
            net = torch.tanh(net)
        return net

    def get_code_indices(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.generator.get_code_indices(net, **kwargs)

    def get_code(self, code_indices: Tensor) -> Tensor:
        return self.generator.get_code(code_indices)

    def reconstruct_from(
        self,
        code_indices: Tensor,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
        use_one_hot: bool = False,
        apply_tanh: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tensor:
        net = self.generator.reconstruct_from(
            code_indices,
            class_idx=class_idx,
            labels=labels,
            use_one_hot=use_one_hot,
            **kwargs,
        )
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        if apply_tanh:
            net = torch.tanh(net)
        return net

    def sample_codebook(
        self,
        *,
        code_indices: Optional[Tensor] = None,
        num_samples: Optional[int] = None,
        class_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        return self.generator.sample_codebook(
            code_indices=code_indices,
            num_samples=num_samples,
            class_idx=class_idx,
            **kwargs,
        )


__all__ = ["VQVAE"]
