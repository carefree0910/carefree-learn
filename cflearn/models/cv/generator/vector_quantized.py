from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..decoder import DecoderBase
from ..encoder import EncoderBase
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ..vae.vector_quantized import VQCodebook
from ....modules.blocks import Conv2d


@ModelProtocol.register("vq_generator")
class VQGenerator(ModelProtocol):
    def __init__(
        self,
        img_size: int,
        latent_channels: int = 256,
        *,
        encoder: str = "vqgan",
        decoder: str = "vqgan",
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        num_code: int = 16384,
        code_dimension: int = 256,
    ):
        super().__init__()
        if encoder_config is None:
            encoder_config = {}
        if decoder_config is None:
            decoder_config = {}
        encoder_config["img_size"] = img_size
        decoder_config["img_size"] = img_size
        encoder_config["latent_channels"] = latent_channels
        decoder_config["latent_channels"] = latent_channels
        self.encoder = EncoderBase.make(encoder, encoder_config)
        self.decoder = DecoderBase.make(decoder, decoder_config)
        self.codebook = VQCodebook(num_code, code_dimension)
        self.q_conv = Conv2d(latent_channels, code_dimension, kernel_size=1)
        self.post_q_conv = Conv2d(code_dimension, latent_channels, kernel_size=1)

    def encode(self, net: Tensor) -> Tensor:
        net = self.encoder(0, {INPUT_KEY: net})[LATENT_KEY]
        net = self.q_conv(net)
        net, _ = self.codebook(net)
        return net

    def decode(self, z_q: Tensor, *, resize: bool = True) -> Tensor:
        net = self.post_q_conv(z_q)
        net = self.decoder(0, {INPUT_KEY: net}, resize=resize)[PREDICTIONS_KEY]
        return net

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.encoder(batch_idx, batch, state, **kwargs)[LATENT_KEY]
        net = self.q_conv(net)
        z_q, indices = self.codebook(net)
        net = self.post_q_conv(z_q)
        rs = self.decoder(batch_idx, {INPUT_KEY: net}, state, **kwargs)
        rs["indices"] = indices
        return rs


__all__ = ["VQGenerator"]
