from typing import Tuple
from cftool.misc import shallow_copy_dict

from ..common import IGenerator
from ..common import EncoderDecoder
from ...core import HijackConv2d


class IAttentionAutoEncoder(IGenerator):
    enc_double_channels: bool

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        embedding_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        attention_type: str = "spatial",
        apply_tanh: bool = False,
    ):
        super().__init__()
        self.z_size = img_size // 2 ** len(channel_multipliers)
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels
        self.latent_channels = latent_channels
        self.channel_multipliers = channel_multipliers
        self.embedding_channels = embedding_channels
        module_config = dict(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            inner_channels=inner_channels,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            downsample_with_conv=resample_with_conv,
            upsample_with_conv=resample_with_conv,
            attention_type=attention_type,
        )
        enc_config, dec_config = map(shallow_copy_dict, 2 * [module_config])
        enc_scaler = 1 + int(self.enc_double_channels)
        enc_config["latent_channels"] = enc_scaler * latent_channels
        dec_config["apply_tanh"] = apply_tanh
        encoder = decoder = "attention"
        self.generator = EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            encoder_config=enc_config,
            decoder_config=dec_config,
        )
        self.to_embedding = HijackConv2d(
            enc_scaler * latent_channels,
            enc_scaler * embedding_channels,
            1,
        )
        self.from_embedding = HijackConv2d(embedding_channels, latent_channels, 1)


__all__ = [
    "IAttentionAutoEncoder",
]
