from torch import Tensor
from typing import Tuple
from typing import Optional

from .common import IAttentionAutoEncoder
from ..common import VQCodebook
from ..common import VQCodebookOut
from ..common import DecoderInputs
from ...common import register_module


@register_module("ae_vq")
class AttentionAutoEncoderVQ(IAttentionAutoEncoder):
    enc_double_channels = False

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        num_codes: int,
        embedding_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        attention_type: str = "spatial",
        apply_tanh: bool = False,
    ):
        super().__init__(
            img_size,
            in_channels,
            out_channels,
            inner_channels,
            latent_channels,
            channel_multipliers,
            embedding_channels=embedding_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
            attention_type=attention_type,
            apply_tanh=apply_tanh,
        )
        self.codebook = VQCodebook(num_codes, embedding_channels)

    def generate_z(self, num_samples: int) -> Tensor:
        # To be honest this is not a good implementation because `vq` is not
        # meant to do generations out of nowhere (or, standard normal).
        # However to keep things unified, we still implement this method.
        z = super().generate_z(num_samples)
        z = self.codebook(z, return_z_q_g=False).z_q
        return z

    def encode(self, net: Tensor) -> VQCodebookOut:
        net = self.generator.encoder.encode(net)
        net = self.to_embedding(net)
        out = self.codebook(net, return_z_q_g=True)
        return out

    def decode(self, inputs: DecoderInputs) -> Tensor:
        inputs.z = self.from_embedding(inputs.z)
        net = self.generator.decoder.decode(inputs)
        return net

    def get_results(
        self,
        net: Tensor,
        *,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> Tuple[Tensor, VQCodebookOut]:
        out = self.encode(net)
        inputs = DecoderInputs(z=out.z_q, no_head=no_head, apply_tanh=apply_tanh)
        net = self.decode(inputs)
        return net, out

    def reconstruct(
        self,
        net: Tensor,
        *,
        labels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        return self.get_results(net)[0]

    def forward(
        self,
        net: Tensor,
        *,
        no_head: bool = False,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        net, _ = self.get_results(net, no_head=no_head, apply_tanh=apply_tanh)
        return net


__all__ = [
    "AttentionAutoEncoderVQ",
]
