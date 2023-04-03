import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .common import IAutoEncoder
from .common import AutoEncoderInit
from .common import AutoEncoderLPIPSWithDiscriminator
from ...schemas import CustomTrainStepLoss
from ....schema import IDLModel
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY


class GaussianDistribution:
    key = "distribution"

    def __init__(self, net: Tensor, deterministic: bool = False):
        self.net = net
        self.device = net.device
        self.deterministic = deterministic
        self.mean, log_var = torch.chunk(net, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean)
            self.var = self.std = zeros
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = torch.exp(self.log_var)

    def sample(self) -> Tensor:
        std = self.std * torch.randn_like(self.mean)
        return self.mean + std

    def kl(self, other: Optional["GaussianDistribution"] = None) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.log_var
            + other.log_var,
            dim=[1, 2, 3],
        )

    def nll(self, sample: Tensor, dims: Tuple[int, ...] = (1, 2, 3)) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0], device=self.device)
        return 0.5 * torch.sum(
            np.log(2.0 * np.pi)
            + self.log_var
            + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> Tensor:
        return self.mean


class AutoEncoderKL(AutoEncoderInit):
    enc_double_channels = True

    def encode(self, net: Tensor) -> GaussianDistribution:
        net = self.generator.encode({INPUT_KEY: net})
        net = self.to_embedding(net)
        return GaussianDistribution(net)

    def decode(
        self,
        z: Tensor,
        *,
        resize: bool = True,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        net = self.from_embedding(z)
        kw = dict(resize=resize, apply_tanh=apply_tanh)
        net = self.generator.decode({INPUT_KEY: net}, **kw)  # type: ignore
        return net

    def forward(
        self,
        net: Tensor,
        *,
        sample_posterior: bool = True,
        apply_tanh: Optional[bool] = None,
    ) -> dict:
        distribution = self.encode(net)
        z = distribution.sample() if sample_posterior else distribution.mode()
        net = self.decode(z, apply_tanh=apply_tanh)
        return {PREDICTIONS_KEY: net, GaussianDistribution.key: distribution}


class AutoEncoderKLLoss(AutoEncoderLPIPSWithDiscriminator):
    def __init__(
        self,
        *,
        # kl
        log_var_init: float = 0.0,
        # common
        kl_weight: float = 1.0,
        d_loss: str = "hinge",
        d_loss_start_step: int = 50001,
        d_num_layers: int = 4,
        d_in_channels: int = 3,
        d_start_channels: int = 64,
        d_factor: float = 1.0,
        d_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ):
        super().__init__(
            kl_weight=kl_weight,
            d_loss=d_loss,
            d_loss_start_step=d_loss_start_step,
            d_num_layers=d_num_layers,
            d_in_channels=d_in_channels,
            d_start_channels=d_start_channels,
            d_factor=d_factor,
            d_weight=d_weight,
            perceptual_weight=perceptual_weight,
        )
        self.log_var = nn.Parameter(torch.ones(size=()) * log_var_init)

    def get_generator_loss(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        *,
        step: Optional[int],
        last_layer: nn.Parameter,
        cond: Optional[Tensor] = None,
        nll_weights: Optional[float] = None,
    ) -> CustomTrainStepLoss:
        inputs = batch[INPUT_KEY].contiguous()
        reconstructions = forward_results[PREDICTIONS_KEY].contiguous()
        loss_items = {}
        # recon & nll & kl loss
        recon_loss = torch.abs(inputs - reconstructions)
        with torch.no_grad():
            loss_items["l1"] = recon_loss.mean().item()
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            p_loss = self.perceptual_weight * p_loss
            recon_loss = recon_loss + p_loss
            with torch.no_grad():
                loss_items["perceptual"] = p_loss.mean().item()
        with torch.no_grad():
            loss_items["recon"] = recon_loss.mean().item()
        nll_loss = recon_loss / torch.exp(self.log_var) + self.log_var
        weighted_nll_loss = nll_loss
        if nll_weights is not None:
            weighted_nll_loss = nll_weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = forward_results[GaussianDistribution.key].kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss
        loss_items["kl"] = self.kl_weight * kl_loss.item()
        # check discriminator start
        if step is not None and step < self.d_loss_start_step:
            return CustomTrainStepLoss(loss, loss_items)
        g_loss = self.g_loss(nll_loss, last_layer, loss_items, reconstructions, cond)
        loss = loss + g_loss
        return CustomTrainStepLoss(loss, loss_items)


@IDLModel.register("ae_kl")
class AutoEncoderKLModel(AutoEncoderKL, IAutoEncoder):  # type: ignore
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
        # loss configs
        use_loss: bool = True,
        kl_weight: float = 1.0,
        log_var_init: float = 0.0,
        d_loss: str = "hinge",
        d_loss_start_step: int = 50001,
        d_num_layers: int = 4,
        d_in_channels: int = 3,
        d_start_channels: int = 64,
        d_factor: float = 1.0,
        d_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        # train step configs
        grad_accumulate: int = 1,
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
        )
        if not use_loss:
            self.loss = None
        else:
            self.loss = AutoEncoderKLLoss(
                kl_weight=kl_weight,
                log_var_init=log_var_init,
                d_loss=d_loss,
                d_loss_start_step=d_loss_start_step,
                d_num_layers=d_num_layers,
                d_in_channels=d_in_channels,
                d_start_channels=d_start_channels,
                d_factor=d_factor,
                d_weight=d_weight,
                perceptual_weight=perceptual_weight,
            )
        self.setup(img_size, grad_accumulate, embedding_channels, channel_multipliers)


__all__ = [
    "AutoEncoderKL",
]
