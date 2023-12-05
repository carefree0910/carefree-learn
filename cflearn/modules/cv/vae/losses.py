import torch

import torch.nn.functional as F

from typing import Any
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from ....schema import losses_type
from ....schema import ILoss
from ....schema import TrainerState
from ....losses import register_loss
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY


MU_KEY = "mu"
LOG_VAR_KEY = "log_var"


@register_loss("vae")
class VAELoss(ILoss):
    kld_w: torch.Tensor

    def __init__(
        self,
        reduction: str = "mean",
        *,
        kld_ema: float = 0.999,
        kld_weight: float = 1.0e-3,
    ):
        super().__init__(reduction)
        self.kld_ema = kld_ema
        self.kld_weight = kld_weight
        self.register_buffer("kld_w", torch.tensor([0.0], dtype=torch.float32))

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results, batch, state

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> losses_type:
        # kld ratio
        if state is not None and not state.is_terminate and self.training:
            eta = self.kld_ema
            ratio = (state.step % state.num_step_per_epoch) / state.num_step_per_epoch
            ratio = (1.0 + ratio) * self.kld_weight
            self.kld_w = eta * self.kld_w + (1.0 - eta) * ratio
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu, log_var = map(forward_results.get, [MU_KEY, LOG_VAR_KEY])
        assert mu is not None and log_var is not None
        var = log_var.exp()
        dim = tuple(i for i in range(1, len(mu.shape)))
        kld_losses = -0.5 * torch.sum(1 + log_var - mu**2 - var, dim=dim)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        if self.training:
            kld_loss = self.kld_w * kld_loss
        loss = mse + kld_loss
        return {"mse": mse, "kld": kld_loss, "kld_w": self.kld_w, LOSS_KEY: loss}


__all__ = [
    "MU_KEY",
    "LOG_VAR_KEY",
    "VAELoss",
]
