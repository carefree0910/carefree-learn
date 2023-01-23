import torch
import random

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from cftool.types import tensor_dict_type

from .discriminators import DiscriminatorBase
from ....schema import _forward
from ....schema import ITrainer
from ....schema import TrainerState
from ....schema import MetricsOutputs
from ....schema import WithDeviceMixin
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ...schemas.cv import GaussianGeneratorMixin
from ....misc.toolkit import mode_context
from ....misc.internal_ import CustomTrainStep
from ....misc.internal_ import CustomTrainStepLoss
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
        self._g_losses(batch, _forward(self, batch_idx, batch, INPUT_KEY))  # type: ignore


class GeneratorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "OneStageGANMixin",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        g_losses = m._g_losses(batch, forward_results)
        g_loss = g_losses.pop(LOSS_KEY)
        g_loss_dict = {k: v.item() for k, v in g_losses.items()}
        g_loss_dict["g"] = g_loss.item()
        return CustomTrainStepLoss(g_loss, g_loss_dict)


class DiscriminatorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "OneStageGANMixin",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        with torch.no_grad():
            detached_forward = {k: v.detach() for k, v in forward_results.items()}
        d_losses = m._d_losses(batch, detached_forward)
        d_loss = d_losses.pop(LOSS_KEY)
        d_loss_dict = {k: v.item() for k, v in d_losses.items()}
        d_loss_dict["g"] = d_loss.item()
        return CustomTrainStepLoss(d_loss, d_loss_dict)


# This mixin should be used with `CustomModule` & `register_custom_module`
class OneStageGANMixin(GANMixin, WithDeviceMixin, metaclass=ABCMeta):
    @property
    def train_steps(self) -> List[CustomTrainStep]:
        return [
            GeneratorStep("core.g_parameters"),
            DiscriminatorStep("core.d_parameters"),
        ]

    def evaluate_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
    ) -> MetricsOutputs:
        forward = _forward(self, batch_idx, batch, INPUT_KEY, state)
        g_losses = self._g_losses(batch, forward)
        # in evaluate step, all tensors are already detached
        d_losses = self._d_losses(batch, forward)
        g_loss = g_losses.pop(LOSS_KEY).item()
        d_loss = d_losses.pop(LOSS_KEY).item()
        loss = g_loss + d_loss
        loss_dict = {"g": g_loss, "d": d_loss, LOSS_KEY: loss}
        loss_dict.update({k: v.item() for k, v in g_losses.items()})
        loss_dict.update({k: v.item() for k, v in d_losses.items()})
        score = weighted_loss_score_fn(loss_dict)
        return MetricsOutputs(score, loss_dict)


class VanillaGANMixin(OneStageGANMixin, GaussianGeneratorMixin, metaclass=ABCMeta):
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
