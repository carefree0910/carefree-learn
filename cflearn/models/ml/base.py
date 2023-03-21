import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional
from typing import NamedTuple

from .encoders import Encoder
from .encoders import EncodingResult
from ...schema import IMLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings


class MLEncodePack(NamedTuple):
    one_hot: Optional[Tensor]
    embedding: Optional[Tensor]
    numerical: Optional[Tensor]
    merged_categorical: Optional[Tensor]
    merged_all: Tensor


class MLModel(IMLModel):
    def __init__(
        self,
        *args: Any,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
        **kwargs: Any
    ):
        super().__init__()
        if encoder_settings is None:
            self.encoder = None
        else:
            self.encoder = Encoder(encoder_settings, global_encoder_settings)

    def encode(self, net: Tensor) -> MLEncodePack:
        if self.encoder is None or self.encoder.is_empty:
            return MLEncodePack(None, None, net, None, net)
        numerical_columns = [
            index
            for index in range(net.shape[-1])
            if index not in self.encoder.tgt_columns
        ]
        numerical = net[..., numerical_columns]
        res: EncodingResult = self.encoder(net)
        merged_categorical = res.merged
        if merged_categorical is None:
            merged_all = numerical
        else:
            merged_all = torch.cat([numerical, merged_categorical], dim=-1)
        return MLEncodePack(
            res.one_hot,
            res.embedding,
            numerical,
            merged_categorical,
            merged_all,
        )


__all__ = [
    "MLModel",
]
