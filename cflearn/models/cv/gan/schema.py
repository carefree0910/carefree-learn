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
from typing import Callable
from typing import Optional
from cftool.types import tensor_dict_type

from ...schemas import CustomTrainStep
from ...schemas import CustomTrainStepLoss
from ...schemas import ModelWithCustomSteps
from ....schema import TrainerState
from ....schema import MetricsOutputs
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ...schemas.cv import GaussianGeneratorMixin
from ....misc.toolkit import get_device
from ....misc.toolkit import mode_context
from ....losses.gan import GANLoss
from ....losses.gan import GANTarget
from ....losses.gan import DiscriminatorOutput


class GeneratorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "IOneStageGAN",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        g_losses = m.g_losses(batch, forward_results)
        g_loss = g_losses.pop(LOSS_KEY)
        g_loss_dict = {k: v.item() for k, v in g_losses.items()}
        g_loss_dict["g"] = g_loss.item()
        return CustomTrainStepLoss(g_loss, g_loss_dict)


class DiscriminatorStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "IOneStageGAN",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        with torch.no_grad():
            detached_forward = {k: v.detach() for k, v in forward_results.items()}
        d_losses = m.d_losses(batch, detached_forward)
        d_loss = d_losses.pop(LOSS_KEY)
        d_loss_dict = {k: v.item() for k, v in d_losses.items()}
        d_loss_dict["g"] = d_loss.item()
        return CustomTrainStepLoss(d_loss, d_loss_dict)


class IOneStageGAN(ModelWithCustomSteps, metaclass=ABCMeta):
    # abstract

    @property
    @abstractmethod
    def g_parameters(self) -> List[nn.Parameter]:
        pass

    @property
    @abstractmethod
    def d_parameters(self) -> List[nn.Parameter]:
        pass

    @abstractmethod
    def g_losses(
        self,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
    ) -> tensor_dict_type:
        pass

    @abstractmethod
    def d_losses(
        self,
        batch: tensor_dict_type,
        detached_forward: tensor_dict_type,
    ) -> tensor_dict_type:
        pass

    # inheritance

    @property
    def train_steps(self) -> List[CustomTrainStep]:
        return [
            GeneratorStep("g_parameters"),
            DiscriminatorStep("d_parameters"),
        ]

    def evaluate(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"],
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        forward_kwargs: Dict[str, Any],
    ) -> MetricsOutputs:
        forward = self.run(batch_idx, batch)
        g_losses = self.g_losses(batch, forward)
        # in evaluate step, all tensors are already detached
        d_losses = self.d_losses(batch, forward)
        g_loss = g_losses.pop(LOSS_KEY).item()
        d_loss = d_losses.pop(LOSS_KEY).item()
        loss = g_loss + d_loss
        loss_dict = {"g": g_loss, "d": d_loss, LOSS_KEY: loss}
        loss_dict.update({k: v.item() for k, v in g_losses.items()})
        loss_dict.update({k: v.item() for k, v in d_losses.items()})
        score = weighted_loss_score_fn(loss_dict)
        return MetricsOutputs(score, loss_dict, {k: False for k in loss_dict})

    def summary_forward(self, batch: tensor_dict_type) -> None:
        self.g_losses(batch, self.run(0, batch))

    # api

    def build_loss(
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


class IGenerator(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, batch: tensor_dict_type) -> Tensor:
        pass


class IDiscriminator(nn.Module):
    @abstractmethod
    def forward(self, net: Tensor) -> DiscriminatorOutput:
        pass


class IVanillaGAN(IOneStageGAN, GaussianGeneratorMixin, metaclass=ABCMeta):
    generator: IGenerator
    discriminator: IDiscriminator

    # inheritance

    @property
    def can_reconstruct(self) -> bool:
        return False

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return batch[INPUT_KEY], batch.get(LABEL_KEY)

    def forward(self, net: Tensor, labels: Optional[Tensor]) -> Tensor:
        z = torch.randn(len(net), self.latent_dim, device=get_device(self))
        return self.decode(z, labels=labels)

    def decode(self, z: Tensor, *, labels: Optional[Tensor], **kwargs: Any) -> Tensor:
        batch = {INPUT_KEY: z, LABEL_KEY: labels}
        net = self.generator.decode(batch, **kwargs)
        return net

    def g_losses(
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

    def d_losses(
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
    "IOneStageGAN",
    "IVanillaGAN",
]
