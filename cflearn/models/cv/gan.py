import torch
import random

from torch import nn
from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from typing import NamedTuple
from cftool.types import tensor_dict_type

from ...schema import DLConfig
from ...schema import IDLModel
from ...schema import TrainStep
from ...schema import TrainerState
from ...schema import TrainStepLoss
from ...modules import build_generator
from ...modules import IDecoder
from ...modules import DecoderInputs
from ...modules import IDiscriminator
from ...modules import DiscriminatorOutput
from ...toolkit import get_gradient
from ...toolkit import mode_context
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY


class GANTarget(NamedTuple):
    target_is_real: bool
    labels: Optional[Tensor] = None


class GradientNormLoss(nn.Module):
    def __init__(self, k: float = 1.0):
        super().__init__()
        self.k = k

    def forward(self, net_input: Tensor, output: Tensor) -> Tensor:
        gradients = get_gradient(output, net_input, True, True)
        gradients = gradients.view(net_input.shape[0], -1)  # type: ignore
        gradients_norm = gradients.norm(2, dim=1)
        return torch.mean((gradients_norm - self.k) ** 2)


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str):
        super().__init__()
        self.loss: nn.Module
        self.gan_mode = gan_mode
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = GradientNormLoss(k=1.0)
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")
        self.ce = nn.CrossEntropyLoss()

    def expand_target(self, tensor: Tensor, target_is_real: bool) -> Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(tensor)  # type: ignore

    def forward(self, output: DiscriminatorOutput, target: GANTarget) -> Tensor:
        predictions, target_is_real = output.output, target.target_is_real
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.expand_target(predictions, target_is_real)
            loss = self.loss(predictions, target_tensor)
        elif self.gan_mode == "wgangp":
            loss = -predictions.mean() if target_is_real else predictions.mean()
        else:
            raise NotImplementedError(f"gan_mode '{self.gan_mode}' is not implemented")
        if output.cond_logits is not None and target.target_is_real:
            cond_loss = self.ce(output.cond_logits, target.labels)
            loss = loss + cond_loss
        return loss


class VanillaGeneratorStep(TrainStep):
    def loss_fn(
        self,
        m: "GANModel",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled = forward_results[PREDICTIONS_KEY]
        pred_fake = m.discriminator(sampled)
        loss_g = m.gan_loss(pred_fake, GANTarget(True, labels))
        return TrainStepLoss(loss_g, {"g": loss_g.item()})


class VanillaDiscriminatorStep(TrainStep):
    def loss_fn(
        self,
        m: "GANModel",
        state: Optional[TrainerState],
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> TrainStepLoss:
        with torch.no_grad():
            detached_forward = {k: v.detach() for k, v in forward_results.items()}
        net = batch[INPUT_KEY]
        labels = batch.get(LABEL_KEY)
        if labels is not None:
            labels = labels.view(-1)
        sampled_tensor = detached_forward[PREDICTIONS_KEY]
        pred_real = m.discriminator(net)
        loss_d_real = m.gan_loss(pred_real, GANTarget(True, labels))
        pred_fake = m.discriminator(sampled_tensor)
        loss_d_fake = m.gan_loss(pred_fake, GANTarget(False, labels))
        d_loss = 0.5 * (loss_d_fake + loss_d_real)
        losses = {"d_fake": loss_d_fake.item(), "d_real": loss_d_real.item()}
        if m.gan_loss.gan_mode == "wgangp":
            eps = random.random()
            merged = eps * net + (1.0 - eps) * sampled_tensor
            with mode_context(m.discriminator, to_train=None, use_grad=True):
                pred_merged = m.discriminator(merged.requires_grad_(True)).output  # type: ignore
                loss_gp = m.gan_loss.loss(merged, pred_merged)
            d_loss = d_loss + m.lambda_gp * loss_gp
            losses["d_gp"] = loss_gp.item()
        losses["d"] = d_loss.item()
        return TrainStepLoss(d_loss, losses)


@IDLModel.register("gan")
class GANModel(IDLModel):
    gan_loss: GANLoss
    lambda_gp: float

    # inheritance

    @property
    def train_steps(self) -> List[TrainStep]:
        return [
            VanillaGeneratorStep("generator"),
            VanillaDiscriminatorStep("discriminator"),
        ]

    @property
    def all_modules(self) -> List[nn.Module]:
        return [self.m, self.gan_loss]

    def build(self, config: DLConfig) -> None:
        self.m = build_generator(config.module_name, config=config.module_config)
        loss_config = config.loss_config or {}
        gan_mode = loss_config.setdefault("gan_mode", "vanilla")
        self.lambda_gp = loss_config.setdefault("lambda_gp", 10.0)
        self.gan_loss = GANLoss(gan_mode)

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        z = self.m.generate_z(len(batch[INPUT_KEY]))
        inputs = DecoderInputs(z=z, labels=batch.get(LABEL_KEY))
        return (inputs,)

    # api

    @property
    def generator(self) -> IDecoder:
        return self.m.generator

    @property
    def discriminator(self) -> IDiscriminator:
        return self.m.discriminator


__all__ = [
    "GANTarget",
    "GradientNormLoss",
    "GANLoss",
    "GANModel",
]
