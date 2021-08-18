import torch

from typing import Any
from typing import Optional

from .protocol import Encoder1DBase
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import WARNING_PREFIX
from ....modules.blocks import Conv2d


@Encoder1DBase.register("dino")
class DINOEncoder(Encoder1DBase):
    def __init__(
        self,
        in_channels: int,
        img_size: Optional[int] = None,
        latent_dim: Optional[int] = None,
        *,
        name: str = "dino_vits8",
        **dino_kwargs: Any,
    ):
        super().__init__(-1, in_channels, -1)
        if img_size is not None:
            print(f"{WARNING_PREFIX}`img_size` will not affect `DINOEncoder`")
        self.dino = torch.hub.load("facebookresearch/dino:main", name, **dino_kwargs)
        if latent_dim is not None and latent_dim != self.dino.num_features:
            raise ValueError(f"`latent_dim` should be {self.dino.num_features}")
        self.to_rgb = Conv2d(in_channels, 3, kernel_size=1, bias=False)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return {LATENT_KEY: self.dino(self.to_rgb(batch[INPUT_KEY]))}


__all__ = ["DINOEncoder"]
