import torch
import random

import torch.nn as nn

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from cftool.array import to_device
from cftool.types import tensor_dict_type

from .discriminators import DiscriminatorBase
from ....data import CVLoader
from ....protocol import _forward
from ....protocol import StepOutputs
from ....protocol import MetricsOutputs
from ....protocol import WithDeviceMixin
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ...protocols.cv import GaussianGeneratorMixin
from ....misc.toolkit import mode_context
from ....misc.toolkit import toggle_optimizer
from ....losses.gan import GANLoss
from ....losses.gan import GANTarget


class GANMixin:
    def _initialize(
        self,
        *,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        forward: tensor_dict_type,
    ) -> tensor_dict_type:
        pass

    @abstractmethod
    def _d_losses(
        self,
        batch: tensor_dict_type,
        detached_forward: tensor_dict_type,
    ) -> tensor_dict_type:
        pass

    # utilities

    @property
    def can_reconstruct(self) -> bool:
        return False

    def summary_forward(self, batch_idx: int, batch: tensor_dict_type) -> None:
        self._g_losses(batch, self.forward(batch_idx, batch))  # type: ignore


class OneStageGANMixin(GANMixin, WithDeviceMixin):
    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        trainer: Any,
        forward_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        opt_g = trainer.optimizers["core.g_parameters"]
        opt_d = trainer.optimizers["core.d_parameters"]
        # forward
        forward = _forward(
            self,
            batch_idx,
            batch,
            INPUT_KEY,
            trainer.state,
            **forward_kwargs,
        )
        with torch.no_grad():
            detached_forward = {k: v.detach() for k, v in forward.items()}
        # generator step
        with toggle_optimizer(self, opt_g):
            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                g_losses = self._g_losses(batch, forward)
            g_loss = g_losses.pop(LOSS_KEY)
            trainer.grad_scaler.scale(g_loss).backward()
            trainer.clip_norm_step()
            trainer.grad_scaler.step(opt_g)
            trainer.grad_scaler.update()
            opt_g.zero_grad()
        # discriminator step
        with toggle_optimizer(self, opt_d):
            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                d_losses = self._d_losses(batch, detached_forward)
            d_loss = d_losses.pop(LOSS_KEY)
            trainer.grad_scaler.scale(d_loss).backward()
            trainer.clip_norm_step()
            trainer.grad_scaler.step(opt_d)
            trainer.grad_scaler.update()
            opt_d.zero_grad()
        # finalize
        trainer.scheduler_step()
        loss_dict = {"g": g_loss.item(), "d": d_loss.item()}
        loss_dict.update({k: v.item() for k, v in g_losses.items()})
        loss_dict.update({k: v.item() for k, v in d_losses.items()})
        return StepOutputs(detached_forward, loss_dict)

    def evaluate_step(
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
            forward = _forward(self, i, batch, INPUT_KEY, trainer.state)
            g_losses = self._g_losses(batch, forward)
            # in evaluate step, all tensors are already detached
            d_losses = self._d_losses(batch, forward)
            g_loss = g_losses.pop(LOSS_KEY).item()
            d_loss = d_losses.pop(LOSS_KEY).item()
            loss = g_loss + d_loss
            loss_dict = {"g": g_loss, "d": d_loss, LOSS_KEY: loss}
            loss_dict.update({k: v.item() for k, v in g_losses.items()})
            loss_dict.update({k: v.item() for k, v in d_losses.items()})
            for k, v in loss_dict.items():
                loss_items.setdefault(k, []).append(v)
        # gather
        mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
        score = trainer.weighted_loss_score(mean_loss_items)
        return MetricsOutputs(score, mean_loss_items)


class VanillaGANMixin(OneStageGANMixin, GaussianGeneratorMixin):
    def _initialize(  # type: ignore
        self,
        *,
        in_channels: int,
        discriminator: str = "basic",
        discriminator_config: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
    ):
        super()._initialize(
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

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> torch.Tensor:
        z = torch.randn(len(batch[INPUT_KEY]), self.latent_dim, device=self.device)
        return self.decode(z, labels=batch[LABEL_KEY], **kwargs)

    def _g_losses(
        self,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
    ) -> tensor_dict_type:
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled = forward[PREDICTIONS_KEY]
        pred_fake = self.discriminator(sampled)
        loss_g = self.gan_loss(pred_fake, GANTarget(True, labels))
        return {LOSS_KEY: loss_g}

    def _d_losses(
        self,
        batch: tensor_dict_type,
        detached_forward: tensor_dict_type,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled_tensor = detached_forward[PREDICTIONS_KEY]
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
