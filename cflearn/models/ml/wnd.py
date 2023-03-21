import torch

from torch import Tensor
from typing import Dict
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from .base import MLModel
from .fcnn import FCNN
from .linear import Linear
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings


@MLModel.register("wnd")
class WideAndDeep(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        hidden_units: Optional[List[int]] = None,
        *,
        mapping_type: str = "basic",
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        encoder = self.encoder
        if encoder is None or encoder.is_empty:
            wide_dim = deep_dim = input_dim
        else:
            wide_dim = encoder.categorical_dim
            numerical_dim = input_dim - encoder.num_one_hot - encoder.num_embedding
            deep_dim = numerical_dim + encoder.embedding_dim
        self.fcnn = FCNN(
            deep_dim,
            output_dim,
            num_history,
            hidden_units,
            mapping_type=mapping_type,
            bias=bias,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        self.linear = Linear(wide_dim, output_dim, num_history, bias=bias)

    def forward(self, net: Tensor) -> tensor_dict_type:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        encoded = self.encode(net)
        one_hot = encoded.one_hot
        embedding = encoded.embedding
        numerical = encoded.numerical
        # wide
        if one_hot is None and embedding is None:
            wide_inp = numerical
        else:
            if one_hot is None:
                wide_inp = embedding
            elif embedding is None:
                wide_inp = one_hot
            else:
                wide_inp = torch.cat([one_hot, embedding], dim=-1)
        wide_net = self.linear(wide_inp)
        # deep
        if embedding is None:
            deep_inp = numerical
        elif numerical is None:
            deep_inp = embedding
        else:
            deep_inp = torch.cat([numerical, embedding], dim=-1)
        deep_net = self.fcnn(deep_inp)
        # merge
        return wide_net + deep_net


__all__ = ["WideAndDeep"]
