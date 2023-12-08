import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from torch.autograd import grad
from cftool.misc import update_dict
from cftool.types import tensor_dict_type

from ..common import build_loss
from ...schema import DLConfig
from ...schema import IDLModel
from ...schema import TrainStep
from ...schema import TrainerState
from ...schema import TrainStepLoss
from ...losses import register_loss
from ...losses import LPIPS
from ...modules import build_module
from ...modules import VQVAELoss
from ...modules import NLayerDiscriminator
from ...modules import IAttentionAutoEncoder
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY
from ...modules.cv.ae.kl import GaussianDistribution
from ...modules.cv.ae.vq import AttentionAutoEncoderVQ


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


class AutoEncoderLPIPSWithDiscriminator(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        *,
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
        super().__init__()
        assert d_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
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
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1.0e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1.0e4).detach()
        d_weight = d_weight * self.d_weight
        return d_weight

    def g_loss(
        self,
        nll_loss: Tensor,
        last_layer: nn.Parameter,
        loss_items: Dict[str, float],
        reconstructions: Tensor,
        cond: Optional[Tensor],
    ) -> Tensor:
        device = nll_loss.device
        if cond is None:
            fake = self.discriminator(reconstructions).output
        else:
            fake = self.discriminator(torch.cat((reconstructions, cond), dim=1)).output
        g_loss = -torch.mean(fake)
        if self.d_factor <= 0.0:
            d_weight = torch.tensor(0.0, device=device)
        else:
            try:
                d_weight = self.get_d_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0, device=device)
        loss_items["g_loss"] = g_loss.item()
        return d_weight * self.d_factor * g_loss

    def get_discriminator_loss(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        *,
        step: Optional[int] = None,
        cond: Optional[Tensor] = None,
    ) -> TrainStepLoss:
        if step is not None and step < self.d_loss_start_step:
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
        return TrainStepLoss(loss, {"d_loss": loss.item()})

    @abstractmethod
    def get_generator_loss(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        *,
        last_layer: nn.Parameter,
        step: Optional[int] = None,
        cond: Optional[Tensor] = None,
    ) -> TrainStepLoss:
        pass


@register_loss("ae_kl")
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
        last_layer: nn.Parameter,
        step: Optional[int] = None,
        cond: Optional[Tensor] = None,
        nll_weights: Optional[float] = None,
    ) -> TrainStepLoss:
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
            return TrainStepLoss(loss, loss_items)
        g_loss = self.g_loss(nll_loss, last_layer, loss_items, reconstructions, cond)
        loss = loss + g_loss
        return TrainStepLoss(loss, loss_items)


@register_loss("ae_vq")
class AutoEncoderVQLoss(AutoEncoderLPIPSWithDiscriminator):
    def __init__(
        self,
        *,
        # vq
        lb_vq: float = 0.25,
        lb_recon: float = 1.0,
        lb_commit: float = 1.0,
        vq_loss_type: str = "l1",
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
        self.vq_loss = VQVAELoss(
            lb_vq=lb_vq,
            lb_recon=lb_recon,
            lb_commit=lb_commit,
            loss_type=vq_loss_type,
        )

    def get_generator_loss(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        *,
        last_layer: nn.Parameter,
        step: Optional[int] = None,
        cond: Optional[Tensor] = None,
    ) -> TrainStepLoss:
        inputs = batch[INPUT_KEY].contiguous()
        reconstructions = forward_results[PREDICTIONS_KEY].contiguous()
        # vq & nll loss
        vq_losses = self.vq_loss(forward_results, batch, reduction="none", gather=False)
        ## {"mse": mse, "commit": commit_loss, LOSS_KEY: loss}
        recon_loss = vq_losses[self.vq_loss.loss_type]
        codebook_loss = vq_losses["codebook"].mean()
        with torch.no_grad():
            loss_items = {
                "recon": recon_loss.mean().item(),
                "codebook": codebook_loss.item(),
            }
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            p_loss = self.perceptual_weight * p_loss
            recon_loss = recon_loss + p_loss
            with torch.no_grad():
                loss_items["perceptual"] = p_loss.mean().item()
        nll_loss = torch.mean(recon_loss)
        loss = nll_loss + codebook_loss
        # check discriminator start
        if step is not None and step < self.d_loss_start_step:
            return TrainStepLoss(loss, loss_items)
        g_loss = self.g_loss(nll_loss, last_layer, loss_items, reconstructions, cond)
        loss = loss + g_loss
        return TrainStepLoss(loss, loss_items)


class GeneratorStep(TrainStep):
    def get_default_optimize_settings(self) -> Optional[Dict[str, Any]]:
        return {
            "optimizer": "adam",
            "scheduler": None,
            "optimizer_config": {"lr": 4.5e-6, "betas": [0.5, 0.9]},
            "scheduler_config": {},
        }

    def loss_fn(
        self,
        m: "AEModel",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        return m.loss.get_generator_loss(
            batch,
            forward_results,
            step=None if state is None else state.step,
            last_layer=m.m.generator.decoder.head[-1].weight,  # type: ignore
        )


class DiscriminatorStep(TrainStep):
    def get_default_optimize_settings(self) -> Optional[Dict[str, Any]]:
        return {
            "optimizer": "adam",
            "scheduler": None,
            "optimizer_config": {"lr": 4.5e-6, "betas": [0.5, 0.9]},
            "scheduler_config": {},
        }

    def should_skip(self, m: "AEModel", state: Optional[TrainerState]) -> bool:
        return state is not None and state.step < m.loss.d_loss_start_step

    def loss_fn(
        self,
        m: "AEModel",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        return m.loss.get_discriminator_loss(
            batch[INPUT_KEY],
            forward_results[PREDICTIONS_KEY],
            step=None if state is None else state.step,
        )


@IDLModel.register("ae")
class AEModel(IDLModel):
    m: IAttentionAutoEncoder
    loss: AutoEncoderLPIPSWithDiscriminator

    @property
    def train_steps(self) -> List[TrainStep]:
        return [
            GeneratorStep("ae_parameters"),
            DiscriminatorStep("d_parameters"),
        ]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m, self.loss]

    def from_accelerator(self, m: nn.Module, loss: nn.Module) -> IDLModel:
        cloned: AEModel = AEModel.from_config(self.config.copy())
        cloned.m = m
        cloned.loss = loss
        return cloned

    def build(self, config: DLConfig) -> None:
        if config.loss_name is None:
            raise ValueError("loss name should be provided")
        module_config = config.module_config or {}
        self.m = build_module(config.module_name, config=module_config)
        loss_defaults = dict(
            kl_weight=1.0,
            log_var_init=0.0,
            d_loss="hinge",
            d_loss_start_step=50001,
            d_num_layers=4,
            d_in_channels=module_config.get("in_channels", 3),
            d_start_channels=64,
            d_factor=1.0,
            d_weight=1.0,
            perceptual_weight=1.0,
        )
        loss_config = update_dict(config.loss_config or {}, loss_defaults)
        self.loss = build_loss(config.loss_name, config=loss_config)

    # api

    @property
    def ae_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.m.generator.parameters())
            + list(self.m.to_embedding.parameters())
            + list(self.m.from_embedding.parameters())
        )

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.loss.discriminator.parameters())


@IDLModel.register("ae_vq")
class AEVQModel(AEModel):
    m: AttentionAutoEncoderVQ

    @property
    def ae_parameters(self) -> List[nn.Parameter]:
        return super().ae_parameters + list(self.m.codebook.parameters())


__all__ = [
    "AEModel",
    "AEVQModel",
]
