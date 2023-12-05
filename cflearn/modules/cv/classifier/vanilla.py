from torch import nn
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from cftool.types import tensor_dict_type

from ..common import build_encoder
from ..common import IEncoder
from ...core import Linear
from ...common import register_module
from ....constants import LATENT_KEY
from ....constants import PREDICTIONS_KEY


@register_module("cv_clf")
class VanillaClassifier(nn.Module):
    encoder: IEncoder

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Optional[int] = None,
        latent_dim: int = 128,
        aux_num_classes: Optional[Dict[str, int]] = None,
        *,
        encoder: str = "vanilla_1d",
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        # encoder
        if encoder_config is None:
            encoder_config = {}
        encoder_config.setdefault("img_size", img_size)
        encoder_config.setdefault("in_channels", in_channels)
        encoder_config.setdefault("latent_dim", latent_dim)
        if aux_num_classes is not None:
            encoder_config.setdefault("aux_heads", sorted(aux_num_classes))
        self.encoder = build_encoder(encoder, config=encoder_config)
        # head
        main_head = Linear(latent_dim, num_classes)
        self.head = None
        self.aux_keys = None
        if aux_num_classes is None:
            self.head = main_head
        else:
            heads = {LATENT_KEY: main_head}
            self.aux_keys = []
            for key, n in aux_num_classes.items():
                heads[key] = Linear(latent_dim, n)
                self.aux_keys.append(key)
            self.heads = nn.ModuleDict(heads)

    def forward(self, net: Tensor, *, return_latent: bool = False) -> tensor_dict_type:
        latent = self.encoder.encode(net)
        if return_latent:
            return {LATENT_KEY: latent}
        if self.head is not None:
            return {PREDICTIONS_KEY: self.head(latent)}
        results = {}
        for key, head in self.heads.items():
            predictions = head(latent)
            results[PREDICTIONS_KEY if key == LATENT_KEY else key] = predictions
        return results


__all__ = [
    "VanillaClassifier",
]
