import torch

import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

from .constants import INPUT_B_KEY
from .constants import LABEL_B_KEY
from ....types import losses_type
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import LOSS_KEY
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import WARNING_PREFIX
from ....constants import PREDICTIONS_KEY
from ..gan.losses import GANTarget
from ..gan.protocol import OneStageGANMixin
from ..gan.discriminators import MultiScaleDiscriminator
from ....misc.toolkit import squeeze
from ....misc.toolkit import interpolate
from ....modules.blocks import get_conv_blocks
from ....modules.blocks import Conv2d
from ....modules.blocks import Lambda
from ....modules.blocks import Mapping
from ....modules.blocks import ResidualBlock
from ....modules.blocks import UpsampleConv2d
from ....modules.blocks import AdaptiveInstanceNorm2d


class StyleEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        out_channels: int,
        num_downsample: int,
        *,
        norm_type: Optional[str] = None,
        activation: str = "relu",
    ):
        super().__init__()
        blocks = get_conv_blocks(
            in_channels,
            latent_channels,
            7,
            1,
            norm_type=norm_type,
            activation=activation,
            padding="reflection",
        )
        for i in range(2):
            blocks.extend(
                get_conv_blocks(
                    latent_channels,
                    2 * latent_channels,
                    4,
                    2,
                    norm_type=norm_type,
                    activation=activation,
                    padding="reflection1",
                )
            )
            latent_channels *= 2
        for i in range(num_downsample - 2):
            blocks.extend(
                get_conv_blocks(
                    latent_channels,
                    latent_channels,
                    4,
                    2,
                    norm_type=norm_type,
                    activation=activation,
                    padding="reflection1",
                )
            )
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                Conv2d(
                    latent_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                Lambda(squeeze),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


# TODO : Try ResidualBlockV2


class ContentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_downsample: int,
        num_residual_blocks: int,
        *,
        norm_type: Optional[str] = "instance",
        activation: str = "relu",
    ):
        super().__init__()
        blocks = get_conv_blocks(
            in_channels,
            latent_channels,
            7,
            1,
            norm_type=norm_type,
            activation=activation,
            padding="reflection",
        )
        for _ in range(num_downsample):
            blocks.extend(
                get_conv_blocks(
                    latent_channels,
                    2 * latent_channels,
                    4,
                    2,
                    norm_type=norm_type,
                    activation=activation,
                    padding="reflection1",
                )
            )
            latent_channels *= 2
        for _ in range(num_residual_blocks):
            blocks.append(
                ResidualBlock(
                    latent_channels,
                    0.0,
                    norm_type=norm_type,
                    activation=activation,
                    padding="reflection",
                )
            )
        self.net = nn.Sequential(*blocks)
        self.out_channels = latent_channels

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        num_residual_blocks: int,
        num_upsample: int,
        *,
        res_norm_type: Optional[str] = "adain",
        norm_type: Optional[str] = "layer_norm",
        activation: str = "relu",
    ):
        super().__init__()
        blocks = []
        for _ in range(num_residual_blocks):
            blocks.append(
                ResidualBlock(
                    latent_channels,
                    0.0,
                    norm_type=res_norm_type,
                    activation=activation,
                    padding="reflection",
                )
            )
        for _ in range(num_upsample):
            blocks.extend(
                get_conv_blocks(
                    latent_channels,
                    latent_channels // 2,
                    5,
                    1,
                    norm_type=norm_type,
                    activation=activation,
                    conv_base=UpsampleConv2d,
                    padding="reflection",
                    factor=2,
                )
            )
            latent_channels //= 2
        blocks.extend(
            get_conv_blocks(
                latent_channels,
                out_channels,
                7,
                1,
                norm_type=None,
                activation="tanh",
                padding="reflection",
            )
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


def get_num_adain_params(model: nn.Module) -> int:
    num = 0
    for m in model.modules():
        if isinstance(m, AdaptiveInstanceNorm2d):
            num += 2 * m.dim
    return num


def assign_adain_params(adain_params: Tensor, model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, AdaptiveInstanceNorm2d):
            mean = adain_params[:, : m.dim]
            std = adain_params[:, m.dim : 2 * m.dim]
            m.bias = mean[..., None, None]
            m.weight = std[..., None, None]
            if adain_params.shape[1] > 2 * m.dim:
                adain_params = adain_params[:, 2 * m.dim :]


class MUNITGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        *,
        latent_channels: int = 64,
        style_channels: int = 8,
        style_downsample: int = 4,
        content_downsample: int = 2,
        num_residual_blocks: int = 4,
        num_mapping_blocks: int = 3,
        mapping_latent_dim: int = 256,
    ):
        super().__init__()
        self.style_encoder = StyleEncoder(
            in_channels,
            latent_channels,
            style_channels,
            style_downsample,
        )
        self.content_encoder = ContentEncoder(
            in_channels,
            latent_channels,
            content_downsample,
            num_residual_blocks,
        )
        self.decoder = Decoder(
            self.content_encoder.out_channels,
            in_channels,
            num_residual_blocks,
            content_downsample,
        )
        mapping_blocks = []
        mapping_in_dim = style_channels
        for _ in range(num_mapping_blocks - 1):
            mapping_blocks.append(
                Mapping(
                    mapping_in_dim,
                    mapping_latent_dim,
                    dropout=0.0,
                    batch_norm=False,
                    activation="relu",
                )
            )
            mapping_in_dim = mapping_latent_dim
        mapping_blocks.append(
            Mapping(
                mapping_latent_dim,
                get_num_adain_params(self.decoder),
                dropout=0.0,
                batch_norm=False,
                activation=None,
            )
        )
        self.adain_mlp = nn.Sequential(*mapping_blocks)

    def forward(self, net: Tensor) -> Tensor:
        content, style = self.encode(net)
        reconstruction = self.decode(content, style)
        return interpolate(reconstruction, anchor=net)

    def encode(self, net: Tensor) -> Tuple[Tensor, Tensor]:
        style = self.style_encoder(net)
        content = self.content_encoder(net)
        return content, style

    def decode(self, content: Tensor, style: Tensor) -> Tensor:
        adain_params = self.adain_mlp(style)
        assign_adain_params(adain_params, self.decoder)
        decoded = self.decoder(content)
        return decoded


def _initialize(m: nn.Module) -> None:
    name = m.__class__.__name__
    if name.find("Conv") == 0 or name.find("Linear") == 0:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight.data, a=0.0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


def _initialize_discriminator(m: nn.Module) -> None:
    name = m.__class__.__name__
    if name.find("Conv") == 0 or name.find("Linear") == 0:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


class MUNITStylizerBase(OneStageGANMixin, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: int = 3,
        *,
        gan_mode: str = "lsgan",
        gan_loss_config: Optional[Dict[str, Any]] = None,
        latent_channels: int = 64,
        style_channels: int = 8,
        style_downsample: int = 4,
        content_downsample: int = 2,
        num_residual_blocks: int = 4,
        num_mapping_blocks: int = 3,
        mapping_latent_dim: int = 256,
        gan_weight: float = 1.0,
        recon_weight: float = 10.0,
        style_recon_weight: float = 1.0,
        content_recon_weight: float = 1.0,
    ):
        super().__init__(
            num_classes=None,
            gan_mode=gan_mode,
            gan_loss_config=gan_loss_config,
        )
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.style_channels = style_channels
        self.style_downsample = style_downsample
        self.content_downsample = content_downsample
        self.num_residual_blocks = num_residual_blocks
        self.num_mapping_blocks = num_mapping_blocks
        self.mapping_latent_dim = mapping_latent_dim
        self.loss_weights = {
            "da": gan_weight,
            "db": gan_weight,
            "ga": gan_weight,
            "gb": gan_weight,
            "ab": recon_weight,
            "ba": recon_weight,
            "a_recon": recon_weight,
            "b_recon": recon_weight,
            "sa_recon": style_recon_weight,
            "sb_recon": style_recon_weight,
            "ca_recon": content_recon_weight,
            "cb_recon": content_recon_weight,
        }
        self._build()

    @abstractmethod
    def _build(self) -> None:
        pass

    def _make_generator(self) -> MUNITGenerator:
        return MUNITGenerator(
            self.in_channels,
            latent_channels=self.latent_channels,
            style_channels=self.style_channels,
            style_downsample=self.style_downsample,
            content_downsample=self.content_downsample,
            num_residual_blocks=self.num_residual_blocks,
            num_mapping_blocks=self.num_mapping_blocks,
            mapping_latent_dim=self.mapping_latent_dim,
        )

    def _make_discriminator(self) -> MultiScaleDiscriminator:
        return MultiScaleDiscriminator(self.in_channels)

    def _random_style(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.style_channels, device=self.device)


@MUNITStylizerBase.register("munit_unified")
class MUNITUnifiedStylizer(MUNITStylizerBase):
    def _build(self) -> None:
        self.l1 = nn.L1Loss()
        self.generator = self._make_generator()
        self.discriminator = self._make_discriminator()
        self.apply(_initialize)

    def _gan_loss(self, net: Tensor, is_real: bool) -> Tensor:
        loss = net.new_zeros(1)
        outputs = self.discriminator(net)
        for output in outputs:
            loss += self.gan_loss(output, GANTarget(is_real))
        return loss

    @property
    def g_parameters(self) -> List[nn.Parameter]:
        return list(self.generator.parameters())

    @property
    def d_parameters(self) -> List[nn.Parameter]:
        return list(self.discriminator.parameters())

    def _g_losses(
        self,
        batch: tensor_dict_type,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[tensor_dict_type, tensor_dict_type, Optional[Tensor]]:
        net_a = batch[INPUT_KEY]
        net_b = batch[INPUT_B_KEY]
        net_ab_target = batch[LABEL_KEY]
        net_ba_target = batch[LABEL_B_KEY]
        sa_random = self._random_style(len(net_a))
        sb_random = self._random_style(len(net_b))
        ca, sa = self.generator.encode(net_a)
        cb, sb = self.generator.encode(net_b)
        g_results: tensor_dict_type = {}
        net_ab = self.generator.decode(ca, sb)
        net_ba = self.generator.decode(cb, sa)
        net_a_recon = self.generator.decode(ca, sa)
        net_b_recon = self.generator.decode(cb, sb)
        net_ba_random = g_results["net_ba"] = self.generator.decode(cb, sa_random)
        net_ab_random = g_results["net_ab"] = self.generator.decode(ca, sb_random)
        cb_recon, sa_recon = self.generator.encode(net_ba_random)
        ca_recon, sb_recon = self.generator.encode(net_ab_random)
        g_losses: losses_type = {
            "ga": self._gan_loss(net_ba_random, True),
            "gb": self._gan_loss(net_ab_random, True),
            "ab": self.l1(net_ab, net_ab_target),
            "ba": self.l1(net_ba, net_ba_target),
            "a_recon": self.l1(net_a_recon, net_a),
            "b_recon": self.l1(net_b_recon, net_b),
            "sa_recon": self.l1(sa_recon, sa_random),
            "sb_recon": self.l1(sb_recon, sb_random),
            "ca_recon": self.l1(ca_recon, ca),
            "cb_recon": self.l1(cb_recon, cb),
        }
        g_losses[LOSS_KEY] = sum(v * self.loss_weights[k] for k, v in g_losses.items())
        return g_losses, g_results, None

    def _d_losses(
        self,
        batch: tensor_dict_type,
        sampled: tensor_dict_type,
        labels: Optional[Tensor],
    ) -> tensor_dict_type:
        net_a = batch[INPUT_KEY]
        net_b = batch[INPUT_B_KEY]
        net_ba = sampled["net_ba"]
        net_ab = sampled["net_ab"]
        loss_da = self._gan_loss(net_ba.detach(), False) + self._gan_loss(net_a, True)
        loss_db = self._gan_loss(net_ab.detach(), False) + self._gan_loss(net_b, True)
        d_losses = dict(da=loss_da, db=loss_db)
        d_losses[LOSS_KEY] = sum(v * self.loss_weights[k] for k, v in d_losses.items())
        return d_losses

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        content = self.generator.content_encoder(net)
        style_net = batch.get(INPUT_B_KEY)
        if style_net is None:
            style = self._random_style(len(net))
        else:
            style = self.generator.style_encoder(style_net)
        return {PREDICTIONS_KEY: self.generator.decode(content, style)}

    def decode(
        self,
        z: Tensor,
        *,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        if labels is not None:
            print(f"{WARNING_PREFIX}`labels` will not affect `MUNITUnifiedStylizer`")
        content = kwargs.get("content")
        if content is None:
            raise ValueError("`content` should be provided for `MUNITUnifiedStylizer`")
        content = self.generator.content_encoder(content)
        return self.generator.decode(content, z)


__all__ = [
    "MUNITGenerator",
    "MUNITUnifiedStylizer",
]
