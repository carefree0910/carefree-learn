import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..encoder import Encoder1DBase
from ....types import tensor_dict_type
from ....protocol import ModelProtocol
from ....protocol import TrainerState
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....constants import INFO_PREFIX
from ....constants import PREDICTIONS_KEY
from ...ml.protocol import MERGED_KEY
from ...ml.protocol import MLCoreProtocol


@ModelProtocol.register("clf")
class VanillaClassifier(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Optional[int] = None,
        latent_dim: int = 128,
        *,
        encoder1d: str = "vanilla",
        head: str = "linear",
        encoder1d_configs: Optional[Dict[str, Any]] = None,
        head_configs: Optional[Dict[str, Any]] = None,
        encoder1d_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        # encoder1d
        if encoder1d_configs is None:
            encoder1d_configs = {}
        if encoder1d != "backbone":
            encoder1d_configs["img_size"] = img_size
        encoder1d_configs["in_channels"] = in_channels
        encoder1d_configs["latent_dim"] = latent_dim
        self.encoder1d = Encoder1DBase.make(encoder1d, config=encoder1d_configs)
        if encoder1d_pretrained_path is not None:
            print(
                f"{INFO_PREFIX}loading pretrained encoder1d "
                f"from '{encoder1d_pretrained_path}'"
            )
            d = torch.load(encoder1d_pretrained_path)
            self.encoder1d.load_state_dict(d)
        # head
        if head_configs is None:
            head_configs = {}
        head_configs["in_dim"] = latent_dim
        head_configs["out_dim"] = num_classes
        head_configs["num_history"] = 1
        self.head = MLCoreProtocol.make(head, config=head_configs)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        batch = shallow_copy_dict(batch)
        encoding = self.encoder1d(batch_idx, batch, state, **kwargs)
        batch[MERGED_KEY] = encoding[LATENT_KEY]
        return self.head(batch_idx, batch, state, **kwargs)

    def classify(self, net: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(0, {INPUT_KEY: net}, **kwargs)[PREDICTIONS_KEY]


__all__ = ["VanillaClassifier"]
