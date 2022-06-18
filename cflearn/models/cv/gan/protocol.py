import torch
import random

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

from .discriminators import DiscriminatorBase
from ..protocol import GaussianGeneratorMixin
from ....data import CVLoader
from ....types import tensor_dict_type
from ....protocol import StepOutputs
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....protocol import ModelWithCustomSteps
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import to_device
from ....misc.toolkit import mode_context
from ....misc.toolkit import toggle_optimizer
from ....losses.gan import GANLoss
from ....losses.gan import GANTarget


class GANMixin(ModelWithCustomSteps, GaussianGeneratorMixin, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gan_mode = gan_mode
        self.gan_loss = GANLoss(gan_mode)
        if gan_loss_config is None:
            gan_loss_config = {}
        self.lambda_gp = gan_loss_config.get("lambda_gp", 10.0)

    @property
    @abstractmethod
    def g_parameters(self) -> List[nn.Parameter]:
        pass

    @property
    @abstractmethod
    def d_parameters(self) -> List[nn.Parameter]:
        pass

    @abstractmethod
    def _g_losses(
        self,
        batch: tensor_dict_type,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, tensor_dict_type, Optional[Tensor]]:
        # g_losses, sampled, labels
        pass

    @abstractmethod
    def _d_losses(
        self,
        batch: tensor_dict_type,
        sampled: tensor_dict_type,
        labels: Optional[Tensor],
    ) -> tensor_dict_type:
        # d_losses
        pass

    # utilities

    @property
    def can_reconstruct(self) -> bool:
        return False

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z = torch.randn(len(batch[INPUT_KEY]), self.latent_dim, device=self.device)
        return {PREDICTIONS_KEY: self.decode(z, labels=batch[LABEL_KEY], **kwargs)}

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self._g_losses(batch, {})


class OneStageGANMixin(GANMixin, metaclass=ABCMeta):
    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        opt_g = trainer.optimizers["g_parameters"]
        opt_d = trainer.optimizers["d_parameters"]
        # generator step
        toggle_optimizer(self, opt_g)
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            g_losses, sampled, labels = self._g_losses(batch, forward_kwargs)
        g_loss = g_losses.pop(LOSS_KEY)
        trainer.grad_scaler.scale(g_loss).backward()
        if trainer.clip_norm > 0.0:
            trainer.clip_norm_step()
        trainer.grad_scaler.step(opt_g)
        trainer.grad_scaler.update()
        opt_g.zero_grad()
        # discriminator step
        toggle_optimizer(self, opt_d)
        with torch.no_grad():
            sampled = {k: v.detach().clone() for k, v in sampled.items()}
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            d_losses = self._d_losses(batch, sampled, labels)
        d_loss = d_losses.pop(LOSS_KEY)
        trainer.grad_scaler.scale(d_loss).backward()
        if trainer.clip_norm > 0.0:
            trainer.clip_norm_step()
        trainer.grad_scaler.step(opt_d)
        trainer.grad_scaler.update()
        opt_d.zero_grad()
        # finalize
        trainer.scheduler_step()
        forward_results = {PREDICTIONS_KEY: sampled}
        loss_dict = {"g": g_loss.item(), "d": d_loss.item()}
        loss_dict.update({k: v.item() for k, v in g_losses.items()})
        loss_dict.update({k: v.item() for k, v in d_losses.items()})
        return StepOutputs(forward_results, loss_dict)

    def evaluate_step(  # type: ignore
        self,
        loader: CVLoader,
        portion: float,
        trainer: Any,
    ) -> MetricsOutputs:
        loss_items: Dict[str, List[float]] = {}
        for i, batch in enumerate(loader):
            if i / len(loader) >= portion:
                break
            batch = to_device(batch, self.device)
            g_losses, sampled, labels = self._g_losses(batch, {})
            d_losses = self._d_losses(batch, sampled, labels)
            g_loss = g_losses.pop(LOSS_KEY)
            d_loss = d_losses.pop(LOSS_KEY)
            loss_dict = {"g": g_loss.item(), "d": d_loss.item()}
            loss_dict.update({k: v.item() for k, v in g_losses.items()})
            loss_dict.update({k: v.item() for k, v in d_losses.items()})
            for k, v in loss_dict.items():
                loss_items.setdefault(k, []).append(v)
        # gather
        mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
        mean_loss_items[LOSS_KEY] = sum(mean_loss_items.values())
        score = trainer.weighted_loss_score(mean_loss_items)
        return MetricsOutputs(score, mean_loss_items)


class VanillaGANMixin(OneStageGANMixin, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: int,
        *,
        discriminator: str = "basic",
        discriminator_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            gan_mode=gan_mode,
            gan_loss_config=gan_loss_config,
        )
        if discriminator_config is None:
            discriminator_config = {}
        discriminator_config["in_channels"] = in_channels
        discriminator_config["num_classes"] = num_classes
        self.discriminator = DiscriminatorBase.make(
            discriminator,
            config=discriminator_config,
        )

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def _g_losses(
        self,
        batch: tensor_dict_type,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, tensor_dict_type, Optional[Tensor]]:
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled = self.sample(len(batch[INPUT_KEY]), labels=labels, **forward_kwargs)
        pred_fake = self.discriminator(sampled)
        loss_g = self.gan_loss(pred_fake, GANTarget(True, labels))
        return {LOSS_KEY: loss_g}, {"sampled": sampled}, labels

    def _d_losses(
        self,
        batch: tensor_dict_type,
        sampled: tensor_dict_type,
        labels: Optional[Tensor],
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        sampled_tensor = sampled["sampled"]
        pred_real = self.discriminator(net)
        loss_d_real = self.gan_loss(pred_real, GANTarget(True, labels))
        pred_fake = self.discriminator(sampled_tensor)
        loss_d_fake = self.gan_loss(pred_fake, GANTarget(False, labels))
        d_loss = 0.5 * (loss_d_fake + loss_d_real)
        losses = {"d_fake": loss_d_fake, "d_real": loss_d_real}
        if self.gan_mode == "wgangp":
            eps = random.random()
            merged = eps * net + (1.0 - eps) * sampled_tensor
            with mode_context(self.discriminator, to_train=None, use_grad=True):
                pred_merged = self.discriminator(merged.requires_grad_(True)).output  # type: ignore
                loss_gp = self.gan_loss.loss(merged, pred_merged)
            d_loss = d_loss + self.lambda_gp * loss_gp
            losses["d_gp"] = loss_gp
        losses[LOSS_KEY] = d_loss
        return losses


__all__ = [
    "GANMixin",
    "OneStageGANMixin",
    "VanillaGANMixin",
]
