import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from ..core import Linear
from ..core import MixedStackedEncoder
from ..common import register_module


class MixedStackedModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        latent_dim: int,
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        feedforward_dim_ratio: float = 1.0,
        use_head_token: bool = False,
        head_pooler: Optional[str] = "mean",
        use_positional_encoding: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_history = num_history
        self.to_mixed_stacked = Linear(input_dim, latent_dim)
        self.mixed_stacked = MixedStackedEncoder(
            latent_dim,
            num_history,
            token_mixing_type=token_mixing_type,
            token_mixing_config=token_mixing_config,
            channel_mixing_type=channel_mixing_type,
            channel_mixing_config=channel_mixing_config,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            latent_dim_ratio=feedforward_dim_ratio,
            use_head_token=use_head_token,
            head_pooler=head_pooler,
            use_positional_encoding=use_positional_encoding,
        )
        self.head = Linear(latent_dim, output_dim)

    def forward(self, net: Tensor) -> Tensor:
        net = self.to_mixed_stacked(net)
        net = self.mixed_stacked(net)
        net = self.head(net)
        return net


@register_module("ml_fnet")
class FNet(MixedStackedModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim=latent_dim,
            token_mixing_type="fourier",
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


@register_module("ml_mixer")
class Mixer(MixedStackedModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim=latent_dim,
            token_mixing_type="mlp",
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


@register_module("ml_transformer")
class Transformer(MixedStackedModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        feedforward_dim_ratio: float = 4.0,
        use_head_token: bool = False,
        attention_config: Optional[Dict[str, Any]],
    ):
        attention_config = attention_config or {}
        attention_config.setdefault("bias", False)
        attention_config.setdefault("num_heads", 8)
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim=latent_dim,
            token_mixing_type="attention",
            token_mixing_config=attention_config,
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=feedforward_dim_ratio,
            use_head_token=use_head_token,
            use_positional_encoding=True,
        )


@register_module("ml_pool_former")
class PoolFormer(MixedStackedModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        latent_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
    ):
        super().__init__(
            input_dim,
            output_dim,
            num_history,
            latent_dim=latent_dim,
            token_mixing_type="pool",
            num_layers=num_layers,
            dropout=dropout,
            norm_type=norm_type,
            feedforward_dim_ratio=1.0,
            use_head_token=False,
            use_positional_encoding=False,
        )


__all__ = [
    "MixedStackedModule",
    "FNet",
    "Mixer",
    "Transformer",
    "PoolFormer",
]
