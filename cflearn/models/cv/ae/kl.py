import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from torch.autograd import grad
from cftool.misc import shallow_copy_dict

from ..general import PureEncoderDecoder
from ..gan.discriminators import NLayerDiscriminator
from ..decoder.attn import AttentionDecoder
from ...protocols import GaussianGeneratorMixin
from ....protocol import tensor_dict_type
from ....protocol import ITrainer
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.internal_ import register_custom_module
from ....misc.internal_ import CustomModule
from ....misc.internal_ import CustomTrainStep
from ....misc.internal_ import CustomTrainStepLoss
from ....losses.lpips import LPIPS


class GaussianDistribution:
    key = "distribution"

    def __init__(self, net: Tensor, deterministic: bool = False):
        self.net = net
        self.device = net.device
        self.deterministic = deterministic
        self.mean, log_var = torch.chunk(net, 2, dim=1)
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        if self.deterministic:
            zeros = torch.zeros_like(self.mean).to(device=self.device)
            self.var = self.std = zeros
        else:
            self.std = torch.exp(0.5 * self.log_var)
            self.var = torch.exp(self.log_var)

    def sample(self) -> Tensor:
        std = self.std * torch.randn(self.mean.shape).to(device=self.device)
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


class AutoEncoderKL(nn.Module):
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
        module_config = dict(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            inner_channels=inner_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            downsample_with_conv=resample_with_conv,
            upsample_with_conv=resample_with_conv,
            attention_type=attention_type,
        )
        enc_config, dec_config = map(shallow_copy_dict, 2 * [module_config])
        enc_config["latent_channels"] = 2 * latent_channels
        dec_config["latent_channels"] = latent_channels
        self.generator = PureEncoderDecoder(
            is_1d=False,
            encoder="attention",
            decoder="attention",
            encoder_config=enc_config,
            decoder_config=dec_config,
        )
        self.to_embedding = nn.Conv2d(2 * latent_channels, 2 * embedding_channels, 1)
        self.from_embedding = nn.Conv2d(embedding_channels, latent_channels, 1)
        self.apply_tanh = apply_tanh

    def encode(self, net: Tensor) -> GaussianDistribution:
        net = self.generator.encode({INPUT_KEY: net})
        net = self.to_embedding(net)
        return GaussianDistribution(net)

    def decode(self, z: Tensor, *, apply_tanh: Optional[bool] = None) -> Tensor:
        net = self.from_embedding(z)
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        net = self.generator.decode({INPUT_KEY: net}, apply_tanh=apply_tanh)
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


def d_hinge_loss(real: Tensor, fake: Tensor) -> Tensor:
    l_real = torch.mean(F.relu(1.0 - real))
    l_fake = torch.mean(F.relu(1.0 + fake))
    loss = 0.5 * (l_real + l_fake)
    return loss


def d_vanilla_loss(real: Tensor, fake: Tensor) -> Tensor:
    l_real = torch.mean(F.softplus(-real))
    l_fake = torch.mean(F.softplus(fake))
    loss = 0.5 * (l_real + l_fake)
    return loss


class AutoEncoderKLLoss(nn.Module):
    def __init__(
        self,
        *,
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
    ):
        super().__init__()
        assert d_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.log_var = nn.Parameter(torch.ones(size=()) * log_var_init)
        self.discriminator = NLayerDiscriminator(
            in_channels=d_in_channels,
            num_layers=d_num_layers,
            start_channels=d_start_channels,
        )
        self.d_loss_start_step = d_loss_start_step
        self.d_loss = d_hinge_loss if d_loss == "hinge" else d_vanilla_loss
        self.d_factor = d_factor
        self.d_weight = d_weight

    def get_d_weight(
        self,
        nll_loss: Tensor,
        g_loss: Tensor,
        last_layer: nn.Parameter,
    ) -> Tensor:
        nll_grads = grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.d_weight
        return d_weight

    def get_generator_loss(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        distribution: GaussianDistribution,
        *,
        step: int,
        last_layer: nn.Parameter,
        cond: Optional[Tensor] = None,
        nll_weights: Optional[float] = None,
    ) -> CustomTrainStepLoss:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
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
        kl_loss = distribution.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss
        loss_items["kl"] = self.kl_weight * kl_loss.item()
        # check discriminator start
        if step < self.d_loss_start_step:
            return CustomTrainStepLoss(loss, loss_items)
        # discriminator loss
        if cond is None:
            fake = self.discriminator(reconstructions).output
        else:
            fake = self.discriminator(torch.cat((reconstructions, cond), dim=1)).output
        g_loss = -torch.mean(fake)
        if self.d_factor <= 0.0:
            d_weight = torch.tensor(0.0, device=inputs.device)
        else:
            try:
                d_weight = self.get_d_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=inputs.device)
        # merge
        loss = loss + d_weight * self.d_factor * g_loss
        loss_items["g_loss"] = g_loss.item()
        return CustomTrainStepLoss(loss, loss_items)

    def get_discriminator_loss(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        *,
        step: int,
        cond: Optional[Tensor] = None,
    ) -> CustomTrainStepLoss:
        if step < self.d_loss_start_step:
            raise ValueError(
                "should not call `get_discriminator_loss` because current step "
                f"({step}) is smaller than the `d_loss_start_step` "
                f"({self.d_loss_start_step})"
            )
        inputs = inputs.contiguous().detach()
        reconstructions = reconstructions.contiguous().detach()
        if cond is None:
            real = self.discriminator(inputs).output
            fake = self.discriminator(reconstructions).output
        else:
            real = self.discriminator(torch.cat((inputs, cond), dim=1)).output
            fake = self.discriminator(torch.cat((reconstructions, cond), dim=1)).output
        loss = self.d_factor * self.d_loss(real, fake)
        return CustomTrainStepLoss(loss, {"d_loss": loss.item()})


class GeneratorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "AutoEncoderKLModel",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        return m.loss.get_generator_loss(
            batch[INPUT_KEY],
            forward_results[PREDICTIONS_KEY],
            forward_results[GaussianDistribution.key],
            step=trainer.state.step,
            last_layer=m.generator.decoder.head[-1].weight,  # type: ignore
        )


class DiscriminatorStep(CustomTrainStep):
    def should_skip(self, m: "AutoEncoderKLModel", state: TrainerState) -> bool:
        return state.step < m.loss.d_loss_start_step

    def loss_fn(
        self,
        m: "AutoEncoderKLModel",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        return m.loss.get_discriminator_loss(
            batch[INPUT_KEY],
            forward_results[PREDICTIONS_KEY],
            step=trainer.state.step,
        )


@register_custom_module("ae_kl")
class AutoEncoderKLModel(AutoEncoderKL, CustomModule, GaussianGeneratorMixin):  # type: ignore
    decoder: AttentionDecoder

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
        self.z_size = img_size // 2 ** len(channel_multipliers)
        self.embedding_channels = embedding_channels
        self.grad_accumulate = grad_accumulate

    @property
    def ae_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.generator.parameters())
            + list(self.to_embedding.parameters())
            + list(self.from_embedding.parameters())
        )

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.loss.discriminator.parameters())

    @property
    def can_reconstruct(self) -> bool:
        return True

    @property
    def train_steps(self) -> List[CustomTrainStep]:
        g_scope = "core.ae_parameters"
        d_scope = "core.d_parameters"
        return [
            GeneratorStep(g_scope, grad_accumulate=self.grad_accumulate),
            DiscriminatorStep(
                d_scope,
                grad_accumulate=self.grad_accumulate,
                requires_new_forward=True,
                requires_grad_in_forward=True,
            ),
        ]

    def evaluate_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: ITrainer,
    ) -> MetricsOutputs:
        forward = self.forward(batch[INPUT_KEY])
        args = self, trainer, batch, forward
        loss_items = {}
        g_out = GeneratorStep().loss_fn(*args)
        loss_items.update(g_out.losses)
        if state.step >= self.loss.d_loss_start_step:
            d_out = DiscriminatorStep().loss_fn(*args)
            loss_items.update(d_out.losses)
        score = -loss_items["recon"]
        return MetricsOutputs(score, loss_items)

    def generate_z(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.embedding_channels, self.z_size, self.z_size)
        return z.to(self.device)


__all__ = [
    "AutoEncoderKL",
]
