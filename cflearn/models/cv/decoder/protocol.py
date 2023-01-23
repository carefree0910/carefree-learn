from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from cftool.misc import safe_execute
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister
from cftool.types import tensor_dict_type

from ....schema import _forward
from ....schema import TrainerState
from ....constants import INPUT_KEY
from ....constants import LABEL_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import interpolate
from ....misc.toolkit import auto_num_layers
from ....modules.blocks import ChannelPadding


decoders: Dict[str, Type["DecoderMixin"]] = {}
decoders_1d: Dict[str, Type["Decoder1DMixin"]] = {}


class IDecoder:
    num_upsample: int
    out_channels: int
    img_size: Optional[int]
    num_classes: Optional[int]
    latent_resolution: Optional[int]

    def _initialize(
        self,
        *,
        out_channels: int,
        img_size: Optional[int],
        num_upsample: Optional[int],
        num_classes: Optional[int],
        latent_resolution: Optional[int],
    ) -> None:
        if num_upsample is None:
            fmt = "`{}` should be provided when `num_upsample` is not"
            if img_size is None:
                raise ValueError(fmt.format("img_size"))
            if latent_resolution is None:
                print_warning(
                    f'{fmt.format("latent_resolution")}, '
                    "and 7 will be used as the default `latent_resolution` now"
                )
                latent_resolution = 7
            num_upsample = auto_num_layers(img_size, latent_resolution, None)
        self.num_upsample = num_upsample
        self.out_channels = out_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.latent_resolution = latent_resolution

    @property
    def is_conditional(self) -> bool:
        return self.num_classes is not None

    def resize(self, net: Tensor, *, determinate: bool = False) -> Tensor:
        if self.img_size is None:
            return net
        return interpolate(net, size=self.img_size, determinate=determinate)

    def decode(self, batch: tensor_dict_type, **kwargs: Any) -> Tensor:
        return run_decoder(self, 0, batch, **kwargs)[PREDICTIONS_KEY]


# decode from latent feature map
class DecoderMixin(IDecoder, WithRegister["DecoderMixin"]):
    d = decoders

    latent_channels: int
    cond_channels: int

    def _init_cond(self, *, cond_channels: int = 16) -> None:
        self.cond_channels = cond_channels
        if self.num_classes is None:
            self.cond = None
        else:
            if self.latent_resolution is None:
                msg = "`latent_resolution` should be provided for conditional modeling"
                raise ValueError(msg)
            self.cond = ChannelPadding(
                self.latent_channels,
                cond_channels,
                self.latent_resolution,
                num_classes=self.num_classes,
            )

    def _inject_cond(self, batch: tensor_dict_type) -> tensor_dict_type:
        if self.cond is None:
            return batch
        batch = shallow_copy_dict(batch)
        batch[INPUT_KEY] = self.cond(batch[INPUT_KEY], batch.get(LABEL_KEY))
        return batch


# decode from 1d latent code
class Decoder1DMixin(IDecoder, WithRegister["Decoder1DMixin"]):
    d = decoders_1d

    latent_dim: int


def make_decoder(name: str, config: Dict[str, Any], *, is_1d: bool = False) -> IDecoder:
    base = (Decoder1DMixin if is_1d else DecoderMixin).get(name)  # type: ignore
    return safe_execute(base, config)


def run_decoder(
    decoder: IDecoder,
    batch_idx: int,
    batch: tensor_dict_type,
    state: Optional[TrainerState] = None,
    **kwargs: Any,
) -> tensor_dict_type:
    return _forward(decoder, batch_idx, batch, INPUT_KEY, state, **kwargs)


__all__ = [
    "make_decoder",
    "run_decoder",
    "IDecoder",
    "DecoderMixin",
    "Decoder1DMixin",
]
