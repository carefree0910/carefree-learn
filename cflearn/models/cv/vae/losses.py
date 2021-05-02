import torch

import torch.nn.functional as F

from typing import Any
from typing import Optional

from ....types import losses_type
from ....types import tensor_dict_type
from ....protocol import LossProtocol
from ....protocol import TrainerState
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY


@LossProtocol.register("vae")
class VAELoss(LossProtocol):
    def _init_config(self) -> None:
        self.kld_ema = self.config.setdefault("kld_ema", 0.999)
        self.kld_weight = self.config.setdefault("kld_weight", 1.0e-3)
        self.register_buffer("kld_ratio", torch.tensor([0.0], dtype=torch.float32))

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        # kld ratio
        if state is not None and not state.is_terminate and self.training:
            eta = self.kld_ema
            ratio = (state.step % state.num_step_per_epoch) / state.num_step_per_epoch
            ratio = (1.0 + ratio) * self.kld_weight
            self.kld_ratio = eta * self.kld_ratio + (1.0 - eta) * ratio
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu, log_var = map(forward_results.get, ["mu", "log_var"])
        assert mu is not None and log_var is not None
        var = log_var.exp()
        kld_losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - var, dim=1)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        if self.training:
            kld_loss = self.kld_ratio * kld_loss
        loss = mse + kld_loss
        return {"mse": mse, "kld": kld_loss, LOSS_KEY: loss}


__all__ = [
    "VAELoss",
]
