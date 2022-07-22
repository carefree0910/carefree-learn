import torch

import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import OneStageGANMixin
from .discriminators import DiscriminatorBase
from ..generator import UnetGenerator
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....losses.gan import GANTarget


@OneStageGANMixin.register("pix2pix")
class Pix2Pix(OneStageGANMixin):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        num_downsample: int = 8,
        *,
        start_channels: int = 64,
        norm_type: Optional[str] = "batch",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_dropout: bool = True,
        discriminator: str = "basic",
        discriminator_config: Optional[Dict[str, Any]] = None,
        gan_mode: str = "vanilla",
        gan_loss_config: Optional[Dict[str, Any]] = None,
        lb_l1: float = 100.0,
    ):
        super().__init__(gan_mode=gan_mode, gan_loss_config=gan_loss_config)
        self.generator = UnetGenerator(
            in_channels,
            out_channels,
            num_downsample,
            start_channels=start_channels,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            use_dropout=use_dropout,
        )
        if discriminator_config is None:
            discriminator_config = {}
        discriminator_config["in_channels"] = in_channels + self.generator.out_channels
        self.discriminator = DiscriminatorBase.make(
            discriminator,
            config=discriminator_config,
        )
        # l1 loss
        self.l1_loss = nn.L1Loss()
        self.lb_l1 = lb_l1

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {PREDICTIONS_KEY: self.generator(batch[INPUT_KEY])}

    def _g_losses(
        self,
        batch: tensor_dict_type,
        forward: tensor_dict_type,
    ) -> tensor_dict_type:
        src_domain = batch[INPUT_KEY]
        tgt_domain = batch[LABEL_KEY]
        fake = forward[PREDICTIONS_KEY]
        fake_concat = torch.cat([src_domain, fake], 1)
        pred_fake = self.discriminator(fake_concat)
        loss_g_gan = self.gan_loss(pred_fake, GANTarget(True))
        loss_g_l1 = self.l1_loss(fake, tgt_domain) * self.lb_l1
        loss_g = loss_g_gan + loss_g_l1
        return {LOSS_KEY: loss_g, "g_gan": loss_g_gan, "g_l1": loss_g_l1}

    def _d_losses(
        self,
        batch: tensor_dict_type,
        detached_forward: tensor_dict_type,
    ) -> tensor_dict_type:
        src_domain = batch[INPUT_KEY]
        tgt_domain = batch[LABEL_KEY]
        fake = detached_forward[PREDICTIONS_KEY]
        fake_concat = torch.cat([src_domain, fake], 1)
        pred_fake = self.discriminator(fake_concat)
        loss_d_fake = self.gan_loss(pred_fake, GANTarget(False))
        real_concat = torch.cat([src_domain, tgt_domain], 1)
        pred_real = self.discriminator(real_concat)
        loss_d_real = self.gan_loss(pred_real, GANTarget(True))
        d_loss = 0.5 * (loss_d_fake + loss_d_real)
        return {LOSS_KEY: d_loss, "d_real": loss_d_real, "d_fake": loss_d_fake}


__all__ = ["Pix2Pix"]
