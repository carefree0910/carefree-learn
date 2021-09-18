import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .constants import INPUT_B_KEY
from .constants import LABEL_B_KEY
from ..decoder import StyleGANDecoder
from ..encoder import Encoder1DBase
from ..encoder import Encoder1DFromPatches
from ...bases import CustomLossBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ..vae.losses import VAELoss
from ..vae.vanilla import reparameterize
from ..vae.constants import MU_KEY
from ..vae.constants import LOG_VAR_KEY


@CustomLossBase.register("style_gan_stylizer")
class StyleGANStylizer(CustomLossBase):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 256,
        in_channels: int = 3,
        *,
        encoder1d: str = "vanilla",
        encoder1d_config: Optional[Dict[str, Any]] = None,
        num_downsample: int = 4,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        vae_loss_config: Optional[Dict[str, Any]] = None,
        **block_kwargs: Any,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        if encoder1d_config is None:
            encoder1d_config = {}
        if Encoder1DFromPatches.check_subclass(encoder1d):
            encoder1d_config["img_size"] = img_size
        encoder1d_config["in_channels"] = in_channels
        encoder1d_config["latent_dim"] = latent_dim * 2
        encoder1d_config["num_downsample"] = num_downsample
        self.style_encoder = Encoder1DBase.make(encoder1d, encoder1d_config)
        self.style_decoder = StyleGANDecoder(
            img_size,
            latent_dim,
            self.out_channels,
            channel_base=channel_base,
            channel_max=channel_max,
            num_classes=num_classes,
            conv_clamp=conv_clamp,
            **block_kwargs,
        )
        self.num_ws = self.style_decoder.num_ws
        self.vae_loss = VAELoss(**(vae_loss_config or {}))

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.style_encoder.encode(batch)
        w_mu, w_log_var = net.chunk(2, dim=1)
        w = reparameterize(w_mu, w_log_var)
        ws = w.unsqueeze(1).repeat([1, self.num_ws, 1])
        batch[INPUT_KEY] = ws
        net = self.style_decoder.decode(batch)
        return {
            PREDICTIONS_KEY: net,
            MU_KEY: w_mu,
            LOG_VAR_KEY: w_log_var,
            LATENT_KEY: w,
            "ws": ws,
        }

    def _get_outputs(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net1 = batch[INPUT_KEY]
        rs1 = self(
            batch_idx,
            {INPUT_KEY: net1, LABEL_KEY: batch.get(LABEL_KEY)},
            state,
            **kwargs,
        )
        y1, w1, ws1 = rs1[PREDICTIONS_KEY], rs1[LATENT_KEY], rs1["ws"]
        net2 = batch[INPUT_B_KEY]
        rs2 = self(
            batch_idx,
            {INPUT_KEY: net2, LABEL_KEY: batch.get(LABEL_B_KEY)},
            state,
            **kwargs,
        )
        y2, w2, ws2 = rs2[PREDICTIONS_KEY], rs2[LATENT_KEY], rs2["ws"]
        y12 = self.style_decoder.decode({INPUT_KEY: ws1, LABEL_KEY: batch[LABEL_B_KEY]})
        y21 = self.style_decoder.decode({INPUT_KEY: ws2, LABEL_KEY: batch[LABEL_KEY]})
        return {
            "x1": net1,
            "y1": y1,
            "w1": w1,
            "mu1": rs1[MU_KEY],
            "log_var1": rs1[LOG_VAR_KEY],
            "x2": net2,
            "y2": y2,
            "w2": w2,
            "mu2": rs2[MU_KEY],
            "log_var2": rs2[LOG_VAR_KEY],
            "y12": y12,
            "y21": y21,
            PREDICTIONS_KEY: y1,
        }

    def _get_losses(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, tensor_dict_type]:
        state = trainer.state
        outputs = self._get_outputs(batch_idx, batch, state, **forward_kwargs)
        x1, y1, w1 = map(outputs.get, ["x1", "y1", "w1"])
        x2, y2, w2 = map(outputs.get, ["x2", "y2", "w2"])
        mu1, log_var1 = map(outputs.get, ["mu1", "log_var1"])
        mu2, log_var2 = map(outputs.get, ["mu2", "log_var2"])
        y12, y21 = map(outputs.get, ["y12", "y21"])
        vae_losses1 = self.vae_loss(
            {PREDICTIONS_KEY: y1, MU_KEY: mu1, LOG_VAR_KEY: log_var1},
            {INPUT_KEY: x1},
            state,
            **loss_kwargs,
        )
        vae_losses2 = self.vae_loss(
            {PREDICTIONS_KEY: y2, MU_KEY: mu2, LOG_VAR_KEY: log_var2},
            {INPUT_KEY: x2},
            state,
            **loss_kwargs,
        )
        kld = vae_losses1["kld"] + vae_losses2["kld"]
        recon = vae_losses1["mse"] + vae_losses2["mse"]
        align = (mu1 - mu2).abs().mean() + (log_var1 - log_var2).abs().mean()  # type: ignore
        transfer = F.mse_loss(y12, x2) + F.mse_loss(y21, x1)
        losses = {"kld": kld, "recon": recon, "align": align, "transfer": transfer}
        loss = kld + recon + 1.0e-3 * align + transfer
        losses[LOSS_KEY] = loss
        return outputs, losses


__all__ = [
    "StyleGANStylizer",
]
