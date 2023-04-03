import torch

import torch.nn as nn

from torch import Tensor
from typing import Tuple
from typing import Optional
from cftool.types import tensor_dict_type

from .common import IAutoEncoder
from .common import AutoEncoderInit
from .common import AutoEncoderLPIPSWithDiscriminator
from ..general import VQCodebook
from ...schemas import CustomTrainStepLoss
from ....schema import IDLModel
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....losses.vae import VQVAELoss


class AutoEncoderVQ(AutoEncoderInit):
    enc_double_channels = False

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        num_code: int,
        embedding_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        attention_type: str = "spatial",
        apply_tanh: bool = False,
    ):
        super().__init__(
            img_size,
            in_channels,
            out_channels,
            inner_channels,
            latent_channels,
            channel_multipliers,
            embedding_channels=embedding_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
            attention_type=attention_type,
            apply_tanh=apply_tanh,
        )
        self.codebook = VQCodebook(num_code, embedding_channels)

    def encode(self, net: Tensor) -> Tensor:
        net = self.generator.encode({INPUT_KEY: net})
        net = self.to_embedding(net)
        return net

    def decode(
        self,
        z: Tensor,
        *,
        resize: bool = True,
        apply_codebook: bool = True,
        apply_tanh: Optional[bool] = None,
    ) -> Tensor:
        if apply_codebook:
            z = self.codebook(z).z_q
        net = self.from_embedding(z)
        kw = dict(resize=resize, apply_tanh=apply_tanh)
        net = self.generator.decode({INPUT_KEY: net}, **kw)  # type: ignore
        return net

    def forward(
        self,
        net: Tensor,
        *,
        apply_tanh: Optional[bool] = None,
    ) -> dict:
        net = self.encode(net)
        out = self.codebook(net, return_z_q_g=True)
        net = self.decode(out.z_q, apply_codebook=False, apply_tanh=apply_tanh)
        results = {PREDICTIONS_KEY: net}
        results.update(out.to_dict())
        return results


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
        step: Optional[int],
        last_layer: nn.Parameter,
        cond: Optional[Tensor] = None,
    ) -> CustomTrainStepLoss:
        inputs = batch[INPUT_KEY].contiguous()
        reconstructions = forward_results[PREDICTIONS_KEY].contiguous()
        # vq & nll loss
        vq_losses = self.vq_loss(forward_results, batch, reduction="none", gather=False)
        """{"mse": mse, "commit": commit_loss, LOSS_KEY: loss}"""
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
            return CustomTrainStepLoss(loss, loss_items)
        g_loss = self.g_loss(nll_loss, last_layer, loss_items, reconstructions, cond)
        loss = loss + g_loss
        return CustomTrainStepLoss(loss, loss_items)


@IDLModel.register("ae_vq")
class AutoEncoderVQModel(AutoEncoderVQ, IAutoEncoder):  # type: ignore
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: int,
        inner_channels: int,
        latent_channels: int,
        channel_multipliers: Tuple[int, ...],
        *,
        num_code: int,
        embedding_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        attention_type: str = "spatial",
        # loss configs
        use_loss: bool = True,
        lb_vq: float = 0.25,
        lb_recon: float = 1.0,
        lb_commit: float = 1.0,
        vq_loss_type: str = "l1",
        kl_weight: float = 1.0,
        d_loss: str = "hinge",
        d_loss_start_step: int = 50001,
        d_num_layers: int = 4,
        d_in_channels: int = 3,
        d_start_channels: int = 64,
        d_factor: float = 1.0,
        d_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        # train step configs
        grad_accumulate: int = 1,
    ):
        super().__init__(
            img_size,
            in_channels,
            out_channels,
            inner_channels,
            latent_channels,
            channel_multipliers,
            num_code=num_code,
            embedding_channels=embedding_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            resample_with_conv=resample_with_conv,
            attention_type=attention_type,
        )
        if not use_loss:
            self.loss = None
        else:
            self.loss = AutoEncoderVQLoss(
                lb_vq=lb_vq,
                lb_recon=lb_recon,
                lb_commit=lb_commit,
                vq_loss_type=vq_loss_type,
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
        self.setup(img_size, grad_accumulate, embedding_channels, channel_multipliers)


__all__ = [
    "AutoEncoderVQ",
]
