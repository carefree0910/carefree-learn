import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from cftool.types import tensor_dict_type

from ..types import losses_type
from ..schema import ILoss
from ..schema import TrainerState
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


MU_KEY = "mu"
LOG_VAR_KEY = "log_var"


@ILoss.register("vae")
@ILoss.register("vae1d")
@ILoss.register("vae2d")
@ILoss.register("siren_vae")
@ILoss.register("style_vae")
class VAELoss(ILoss):
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


@ILoss.register("vq_vae")
class VQVAELoss(ILoss):
    def __init__(
        self,
        *,
        lb_vq: float = 1.0,
        lb_recon: float = 1.0,
        lb_commit: float = 1.0,
        loss_type: str = "l2",
    ):
        super().__init__()
        self.lb_vq = lb_vq
        self.lb_recon = lb_recon
        self.lb_commit = lb_commit
        self.loss_type = loss_type

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        *,
        reduction: str = "mean",
        gather: bool = True,
    ) -> losses_type:
        # reconstruction loss
        original = batch[INPUT_KEY]
        reconstruction = forward_results[PREDICTIONS_KEY]
        if self.loss_type == "l2":
            recon = F.mse_loss(reconstruction, original, reduction=reduction)
        elif self.loss_type == "l1":
            recon = F.l1_loss(reconstruction, original, reduction=reduction)
        else:
            raise ValueError(f"unrecognized loss_type '{self.loss_type}' occurred")
        # vq & commit loss
        z_e = forward_results["z_e"]
        z_q_g = forward_results["z_q_g"]
        vq_loss = F.mse_loss(z_q_g, z_e.detach(), reduction=reduction)
        commit_loss = F.mse_loss(z_e, z_q_g.detach(), reduction=reduction)
        codebook_loss = self.lb_vq * vq_loss + self.lb_commit * commit_loss
        losses = {
            self.loss_type: recon,
            "commit": commit_loss,
            "codebook": codebook_loss,
        }
        if not gather:
            return losses
        # gather
        loss = self.lb_recon * recon + codebook_loss
        losses[LOSS_KEY] = loss
        return losses


__all__ = [
    "MU_KEY",
    "LOG_VAR_KEY",
    "VAELoss",
    "VQVAELoss",
]
