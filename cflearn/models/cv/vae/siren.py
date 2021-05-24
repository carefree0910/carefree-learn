from torch import Tensor
from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional

from .vanilla import VanillaVAE
from ..toolkit import auto_num_layers
from ..protocol import GaussianGeneratorMixin
from ....types import tensor_dict_type
from ....trainer import TrainerState
from ....protocol import ModelProtocol
from ....constants import LABEL_KEY
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY
from ..encoder.protocol import Encoder1DBase
from ..implicit.siren import Siren
from ....modules.blocks import Lambda
from ....modules.blocks import Linear
from ....modules.blocks import ChannelPadding


def _siren_head(size: int, out_channels: int) -> Callable[[Tensor], Tensor]:
    def _head(t: Tensor) -> Tensor:
        t = t.view(-1, size, size, out_channels)
        t = t.permute(0, 3, 1, 2)
        return t

    return _head


@ModelProtocol.register("siren_vae")
class SirenVAE(ModelProtocol, GaussianGeneratorMixin):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        min_size: int = 2,
        target_downsample: int = 4,
        latent_dim: int = 256,
        num_classes: Optional[int] = None,
        conditional_dim: int = 16,
        *,
        num_layers: int = 4,
        w_sin: float = 1.0,
        w_sin_initial: float = 30.0,
        bias: bool = True,
        final_activation: Optional[str] = None,
        encoder1d: str = "vanilla",
        encoder1d_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels or in_channels
        num_downsample = auto_num_layers(img_size, min_size, target_downsample)
        # encoder
        if encoder1d_configs is None:
            encoder1d_configs = {}
        encoder1d_configs["img_size"] = img_size
        encoder1d_configs["in_channels"] = in_channels
        encoder1d_configs["latent_dim"] = latent_dim
        if encoder1d == "vanilla":
            encoder1d_configs["num_downsample"] = num_downsample
        self.encoder = Encoder1DBase.make(encoder1d, config=encoder1d_configs)
        self.to_statistics = Linear(latent_dim, 2 * latent_dim, bias=False)
        # condition
        self.cond_padding = None
        if num_classes is not None:
            self.cond_padding = ChannelPadding(conditional_dim, num_classes=num_classes)
            latent_dim += conditional_dim
        # siren
        self.siren = Siren(
            img_size,
            2,
            self.out_channels,
            latent_dim,
            num_layers=num_layers,
            w_sin=w_sin,
            w_sin_initial=w_sin_initial,
            bias=bias,
            final_activation=final_activation,
        )
        # head
        self.head = Lambda(_siren_head(img_size, self.out_channels), name="head")

    @property
    def can_reconstruct(self) -> bool:
        return True

    def decode(
        self,
        z: Tensor,
        *,
        labels: Optional[Tensor],
        size: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        if self.cond_padding is None:
            if labels is not None:
                msg = "`labels` should not be provided in non-conditional `SirenVAE`"
                raise ValueError(msg)
        else:
            if labels is None:
                msg = "`labels` should be provided in conditional `SirenVAE`"
                raise ValueError(msg)
            z = self.cond_padding(z, labels)
        net = self.siren(z, size=size)
        if size is None:
            return self.head(net)
        return _siren_head(size, self.out_channels)(net)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = self.encoder.encode(batch, **kwargs)[LATENT_KEY]
        net = self.to_statistics(net)
        mu, log_var = net.chunk(2, dim=1)
        net = VanillaVAE.reparameterize(mu, log_var)
        if self.cond_padding is not None:
            net = self.cond_padding(net, batch[LABEL_KEY].view(-1))
        net = self.siren(net)
        net = self.head(net)
        return {PREDICTIONS_KEY: net, "mu": mu, "log_var": log_var}


__all__ = [
    "SirenVAE",
]
