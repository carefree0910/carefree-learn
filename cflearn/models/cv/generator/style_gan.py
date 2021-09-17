import torch

from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..protocol import ImageTranslatorMixin
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ..decoder.style_gan import FullyConnected
from ..decoder.style_gan import StyleGANDecoder


def normalize(net: Tensor, dim: int = 1, eps: float = 1.0e-8) -> Tensor:
    return net * (net.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class MappingNetwork(nn.Module):
    def __init__(
        self,
        num_ws: int,
        *,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        num_layers: int = 8,
        embed_features: Optional[int] = None,
        layer_features: Optional[int] = None,
        activation: str = "leaky_relu_0.2",
        lr_multiplier: float = 0.01,
        w_avg_beta: float = 0.995,
    ):
        super().__init__()
        self.num_ws = num_ws
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = latent_dim
        if num_classes is None:
            embed_features = 0
        if layer_features is None:
            layer_features = latent_dim
        features_list = (
            [latent_dim + embed_features]
            + [layer_features] * (num_layers - 1)
            + [latent_dim]
        )

        if num_classes is None:
            self.embed = None
        else:
            self.embed = FullyConnected(num_classes, embed_features)
        blocks = []
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            blocks.append(
                FullyConnected(
                    in_features,
                    out_features,
                    lr_multiplier=lr_multiplier,
                    activation=activation,
                )
            )
        self.net = nn.Sequential(*blocks)
        self.register_buffer("w_avg", torch.zeros([latent_dim]))

    def forward(
        self,
        z: Tensor,
        labels: Optional[Tensor],
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        skip_w_avg_update: bool = False,
    ) -> Tensor:
        net = normalize(z)
        if self.embed is not None:
            if labels is None:
                raise ValueError("`labels` should be provided for conditional mapping")
            y = normalize(self.embed(labels.to(torch.float32)))
            net = torch.cat([net, y], dim=1)
        net = self.net(net)
        if self.training and not skip_w_avg_update:
            updated = net.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            self.w_avg.copy_(updated)
        net = net.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1.0:
            if truncation_cutoff is None:
                net = self.w_avg.lerp(net, truncation_psi)
            else:
                new = self.w_avg.lerp(net[:, :truncation_cutoff], truncation_psi)
                net[:, :truncation_cutoff] = new
        return net


@ModelProtocol.register("style_gan_generator")
class StyleGANGenerator(ImageTranslatorMixin, ModelProtocol):
    def __init__(
        self,
        img_size: int,
        latent_dim: int = 512,
        out_channels: int = 3,
        latent_channels: int = 256,
        *,
        num_layers: int = 8,
        channel_base: int = 32768,
        channel_max: int = 512,
        num_classes: Optional[int] = None,
        conv_clamp: Optional[float] = 256.0,
        block_kwargs: Optional[Dict[str, Any]] = None,
        mapping_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.decoder = StyleGANDecoder(
            img_size,
            latent_dim,
            out_channels,
            latent_channels,
            channel_base=channel_base,
            channel_max=channel_max,
            num_classes=num_classes,
            conv_clamp=conv_clamp,
            **(block_kwargs or {}),
        )
        self.mapping = MappingNetwork(
            self.decoder.num_ws,
            latent_dim=latent_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            **(mapping_kwargs or {}),
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        z = batch[INPUT_KEY]
        labels = batch.get(LABEL_KEY)
        truncation_psi = kwargs.pop("truncation_psi", 1.0)
        truncation_cutoff = kwargs.pop("truncation_cutoff", None)
        ws = self.mapping(z, labels, truncation_psi, truncation_cutoff)
        rgb = self.decoder(batch_idx, {INPUT_KEY: ws}, state, **kwargs)[PREDICTIONS_KEY]
        return {PREDICTIONS_KEY: rgb}


__all__ = ["StyleGANGenerator"]