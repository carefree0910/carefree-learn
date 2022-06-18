import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Optional

from ..types import losses_type
from ..types import tensor_dict_type
from ..protocol import LossProtocol
from ..protocol import TrainerState
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY
from ..misc.internal_ import register_loss_module


MU_KEY = "mu"
LOG_VAR_KEY = "log_var"


@register_loss_module("vae")
@register_loss_module("vae1d")
@register_loss_module("vae2d")
@register_loss_module("siren_vae")
@register_loss_module("style_vae")
class VAELoss(nn.Module):
    kld_w: torch.Tensor

    def __init__(
        self,
        *,
        kld_ema: float = 0.999,
        kld_weight: float = 1.0e-3,
    ):
        super().__init__()
        self.kld_ema = kld_ema
        self.kld_weight = kld_weight
        self.register_buffer("kld_w", torch.tensor([0.0], dtype=torch.float32))

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


@register_loss_module("vq_vae")
class VQVAELoss(nn.Module):
    def __init__(
        self,
        *,
        lb_vq: float = 1.0,
        lb_recon: float = 1.0,
        lb_commit: float = 1.0,
    ):
        super().__init__()
        self.lb_vq = lb_vq
        self.lb_recon = lb_recon
        self.lb_commit = lb_commit

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
    ) -> losses_type:
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # vq & commit loss
        z_e = forward_results["z_e"]
        z_q_g = forward_results["z_q_g"]
        vq_loss = F.mse_loss(z_q_g, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q_g.detach())
        # gather
        loss = self.lb_recon * mse + self.lb_vq * vq_loss + self.lb_commit * commit_loss
        return {"mse": mse, "commit": commit_loss, LOSS_KEY: loss}


__all__ = [
    "MU_KEY",
    "LOG_VAR_KEY",
    "VAELoss",
    "VQVAELoss",
]
