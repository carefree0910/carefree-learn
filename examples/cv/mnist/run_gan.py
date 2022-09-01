import torch
import cflearn

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from cftool.array import to_device
from cftool.types import tensor_dict_type
from cflearn.protocol import MetricsOutputs
from cflearn.constants import INPUT_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import interpolate
from cflearn.misc.toolkit import inject_debug
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import UpsampleConv2d


# preparations
is_ci = check_is_ci()
data = cflearn.cv.MNISTData(batch_size=16, transform="to_tensor")
# for reproduction
np.random.seed(142857)
torch.manual_seed(142857)


class GANLoss(nn.Module):
    def __init__(self):  # type: ignore
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

    def expand_target(self, tensor: Tensor, target_is_real: bool) -> Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(tensor)  # type: ignore

    def forward(self, predictions: Tensor, target_is_real: bool) -> Tensor:
        target_tensor = self.expand_target(predictions, target_is_real)
        loss = self.loss(predictions, target_tensor)
        return loss


class GeneratorStep(cflearn.CustomTrainStep):
    def loss_fn(
        self,
        m: "SimpleGAN",
        trainer: cflearn.ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> cflearn.CustomTrainStepLoss:
        sampled = forward_results[PREDICTIONS_KEY]
        pred_fake = m.discriminator(sampled)
        g_loss = m.loss(pred_fake, target_is_real=True)
        return cflearn.CustomTrainStepLoss(g_loss, {"g": g_loss.item()})


class DiscriminatorStep(cflearn.CustomTrainStep):
    def loss_fn(
        self,
        m: "SimpleGAN",
        trainer: cflearn.ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> cflearn.CustomTrainStepLoss:
        net = batch[INPUT_KEY]
        sampled = forward_results[PREDICTIONS_KEY]
        pred_real = m.discriminator(net)
        loss_d_real = m.loss(pred_real, target_is_real=True)
        pred_fake = m.discriminator(sampled.detach())
        loss_d_fake = m.loss(pred_fake, target_is_real=False)
        d_loss = 0.5 * (loss_d_fake + loss_d_real)
        d_losses = {
            "d": d_loss.item(),
            "d_real": loss_d_real.item(),
            "d_fake": loss_d_fake.item(),
        }
        return cflearn.CustomTrainStepLoss(d_loss, d_losses)


@cflearn.register_custom_module("simple_gan")
class SimpleGAN(cflearn.CustomModule):
    def __init__(self, in_channels: int, img_size: int, latent_dim: int):
        super().__init__()
        if not latent_dim % 16 == 0:
            raise ValueError(f"`latent_dim` ({latent_dim}) should be divided by 16")
        self.latent_dim = latent_dim
        latent_channels = latent_dim // 16
        self.generator = nn.Sequential(
            Lambda(lambda t: t.view(-1, latent_channels, 4, 4), name="reshape"),
            nn.Conv2d(latent_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            UpsampleConv2d(128, 64, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            UpsampleConv2d(64, 32, kernel_size=3, padding=1, factor=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            UpsampleConv2d(32, in_channels, kernel_size=3, padding=1, factor=2),
            Lambda(lambda t: interpolate(t, size=img_size, mode="bilinear")),
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
        )
        self.loss = GANLoss()

    @property
    def train_steps(self) -> List[cflearn.CustomTrainStep]:
        return [
            GeneratorStep("core.g_parameters"),
            DiscriminatorStep("core.d_parameters"),
        ]

    def evaluate_step(  # type: ignore
        self,
        batch: tensor_dict_type,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
    ) -> MetricsOutputs:
        batch = to_device(batch, self.device)
        net = batch[INPUT_KEY]
        sampled = self.sample(len(net))
        pred_fake = self.discriminator(sampled)
        g_loss = self.loss(pred_fake, target_is_real=True)
        pred_real = self.discriminator(net)
        d_loss = self.loss(pred_real, target_is_real=True)
        loss_items = {"g": g_loss.item(), "d": d_loss.item()}
        score = weighted_loss_score_fn(loss_items)
        return MetricsOutputs(score, loss_items)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.generator(z)

    def forward(self, net: Tensor) -> Tensor:  # type: ignore
        return self.sample(len(net))


config = dict(cuda=None if is_ci else 0)
if is_ci:
    inject_debug(config)

cflearn.api.fit_cv(
    data,
    "simple_gan",
    {"in_channels": 1, "img_size": 28, "latent_dim": 128},
    optimizer_settings={
        "core.g_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
        "core.d_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
    },
    **config,  # type: ignore
)
