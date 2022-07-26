import torch
import cflearn

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from cftool.types import tensor_dict_type
from cflearn.types import losses_type
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


@cflearn.register_module("simple_vae")
class SimpleVAE(nn.Module):
    def __init__(self, in_channels: int, img_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
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
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.decoder = nn.Sequential(
            Lambda(lambda t: t.view(-1, 4, 4, 4), name="reshape"),
            nn.Conv2d(4, 128, 1),
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

    def forward(self, net: torch.Tensor) -> Dict[str, torch.Tensor]:
        net = self.encoder(net)
        mu, log_var = net.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        net = eps * std + mu
        net = self.decoder(net)
        return {"mu": mu, "log_var": log_var, cflearn.PREDICTIONS_KEY: net}


@cflearn.register_loss_module("simple_vae")
@cflearn.register_loss_module("simple_vae_foo")
class SimpleVAELoss(nn.Module):
    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
    ) -> losses_type:
        # reconstruction loss
        original = batch[cflearn.INPUT_KEY]
        reconstruction = forward_results[cflearn.PREDICTIONS_KEY]
        mse = F.mse_loss(reconstruction, original)
        # kld loss
        mu = forward_results["mu"]
        log_var = forward_results["log_var"]
        kld_losses = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld_losses, dim=0)
        # gather
        loss = mse + 0.001 * kld_loss
        return {"mse": mse, "kld": kld_loss, cflearn.LOSS_KEY: loss}


config = dict(cuda=None if is_ci else 0)
if is_ci:
    inject_debug(config)

cflearn.api.fit_cv(
    data,
    "simple_vae",
    {"in_channels": 1, "img_size": 28},
    **config,  # type: ignore
)
