import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import PREDICTIONS_KEY
from ..generator.vector_quantized import VQGenerator
from ....misc.toolkit import auto_num_layers


@ModelProtocol.register("vq_vae")
class VQVAE(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        num_code: int,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: Optional[int] = None,
        min_size: int = 4,
        target_downsample: int = 6,
        *,
        encoder: str = "vanilla",
        decoder: str = "vanilla",
        code_dimension: int = 256,
        latent_channels: int = 256,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        latent_padding_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_channels = latent_channels
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
        **kwargs: Any,
    ) -> tensor_dict_type:
        results = self.generator(batch_idx, batch, state, **kwargs)
        results[PREDICTIONS_KEY] = torch.tanh(results[PREDICTIONS_KEY])
        z_q, indices = results["z_q"], results["indices"]
        z_q_g_flatten = self.generator.codebook.embedding.weight[indices]
        shape = -1, *z_q.shape[2:], self.latent_channels
        z_q_g = z_q_g_flatten.view(*shape).permute(0, 3, 1, 2).contiguous()
        results["z_q_g"] = z_q_g
        return results

    def decode(
        self,
        z_q: Tensor,
        *,
        labels: Optional[Tensor] = None,
        resize: bool = True,
    ) -> Tensor:
        net = self.generator.decode(z_q, labels=labels, resize=resize)
        return torch.tanh(net)

    def get_code_indices(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.generator.get_code_indices(net, **kwargs)

    def reconstruct_from(
        self,
        code_indices: Tensor,
        *,
        class_idx: Optional[int] = None,
        labels: Optional[Tensor] = None,
        use_one_hot: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        net = self.generator.reconstruct_from(
            code_indices,
            class_idx=class_idx,
            labels=labels,
            use_one_hot=use_one_hot,
            **kwargs,
        )
        return torch.tanh(net)

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
