import torch

from typing import Dict
from typing import Optional

from .base import MLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings


@MLModel.register("linear")
class Linear(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        bias: bool = True,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        input_dim *= num_history
        self.net = torch.nn.Linear(input_dim, output_dim, bias)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        return self.net(net)


__all__ = ["Linear"]
