import torch
import cflearn

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from torch.optim import Optimizer
from cflearn.types import tensor_dict_type
from cflearn.protocol import StepOutputs
from cflearn.protocol import TrainerState
from cflearn.protocol import MetricsOutputs
from cflearn.protocol import DataLoaderProtocol
from cflearn.constants import INPUT_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import interpolate
from cflearn.misc.toolkit import inject_debug
from cflearn.misc.toolkit import toggle_optimizer
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import UpsampleConv2d
from torch.cuda.amp.grad_scaler import GradScaler


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

    def expand_target(self, tensor: Tensor, use_real_label: bool) -> Tensor:
        target = self.real_label if use_real_label else self.fake_label
        return target.expand_as(tensor)  # type: ignore

    def forward(self, predictions: Tensor, use_real_label: bool) -> Tensor:
        target_tensor = self.expand_target(predictions, use_real_label)
        loss = self.loss(predictions, target_tensor)
        return loss


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

    def train_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        optimizers: Dict[str, Optimizer],
        use_amp: bool,
        grad_scaler: GradScaler,
        clip_norm_fn: Callable[[], None],
        scheduler_step_fn: Callable[[], None],
        trainer: Any,
        forward_kwargs: Dict[str, Any],
        loss_kwargs: Dict[str, Any],
    ) -> StepOutputs:
        net = batch[INPUT_KEY]
        opt_g = optimizers["core.g_parameters"]
        opt_d = optimizers["core.d_parameters"]
        # generator step
        toggle_optimizer(self, opt_g)
        with torch.cuda.amp.autocast(enabled=use_amp):
            sampled = self.sample(len(net))
            pred_fake = self.discriminator(sampled)
            g_loss = self.loss(pred_fake, use_real_label=True)
        grad_scaler.scale(g_loss).backward()
        clip_norm_fn()
        grad_scaler.step(opt_g)
        grad_scaler.update()
        opt_g.zero_grad()
        # discriminator step
        toggle_optimizer(self, opt_d)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_real = self.discriminator(net)
            loss_d_real = self.loss(pred_real, use_real_label=True)
            pred_fake = self.discriminator(sampled.detach().clone())
            loss_d_fake = self.loss(pred_fake, use_real_label=False)
            d_loss = 0.5 * (loss_d_fake + loss_d_real)
        grad_scaler.scale(d_loss).backward()
        clip_norm_fn()
        grad_scaler.step(opt_d)
        grad_scaler.update()
        opt_d.zero_grad()
        # finalize
        scheduler_step_fn()
        forward_results = {PREDICTIONS_KEY: sampled}
        loss_dict = {
            "g": g_loss.item(),
            "d": d_loss.item(),
            "d_fake": loss_d_fake.item(),
            "d_real": loss_d_real.item(),
        }
        return StepOutputs(forward_results, loss_dict)

    def evaluate_step(
        self,
        loader: DataLoaderProtocol,
        portion: float,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: Any,
    ) -> MetricsOutputs:
        loss_items: Dict[str, List[float]] = {}
        for i, batch in enumerate(loader):
            if i / len(loader) >= portion:
                break
            batch = to_device(batch, self.device)
            net = batch[INPUT_KEY]
            sampled = self.sample(len(net))
            pred_fake = self.discriminator(sampled)
            g_loss = self.loss(pred_fake, use_real_label=True)
            pred_real = self.discriminator(net)
            d_loss = self.loss(pred_real, use_real_label=True)
            loss_items.setdefault("g", []).append(g_loss.item())
            loss_items.setdefault("d", []).append(d_loss.item())
        # gather
        mean_loss_items = {k: sum(v) / len(v) for k, v in loss_items.items()}
        mean_loss_items[cflearn.LOSS_KEY] = sum(mean_loss_items.values())
        score = weighted_loss_score_fn(mean_loss_items)
        return MetricsOutputs(score, mean_loss_items)

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.generator(z)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.sample(len(batch[INPUT_KEY]))}


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
