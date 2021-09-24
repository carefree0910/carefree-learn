import torch

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .protocol import Encoder1DFromPatches
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....constants import LATENT_KEY
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("vit")
class ViTEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 384,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        *,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        first_norm: Optional[torch.nn.Module] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        sequence_pool: bool = False,
        use_head_token: bool = True,
        use_positional_encoding: bool = True,
        norm_after_head: bool = False,
        output_dim: Optional[int] = None,
        aux_heads: Optional[List[str]] = None,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            latent_dim,
            to_patches_type,
            to_patches_config,
        )
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", 6)
        self.aux_heads = aux_heads
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            first_norm=first_norm,
            residual_after_norm=residual_after_norm,
            feedforward_dim_ratio=feedforward_dim_ratio,
            sequence_pool=sequence_pool,
            use_head_token=use_head_token,
            use_positional_encoding=use_positional_encoding,
            norm_after_head=norm_after_head,
            aux_heads=aux_heads,
        )
        if output_dim is None:
            self.output_projection = None
        else:
            init = (latent_dim ** -0.5) * torch.randn(latent_dim, output_dim)
            self.output_projection = torch.nn.Parameter(init)

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        rs = super().forward(batch_idx, batch, state, **kwargs)
        latent = rs[LATENT_KEY]
        if self.aux_heads is None:
            if self.output_projection is not None:
                rs[LATENT_KEY] = latent @ self.output_projection
        else:
            keys = self.aux_heads + [LATENT_KEY]
            chunked = latent.chunk(len(self.aux_heads) + 1, dim=1)
            for k, v in zip(keys, chunked):
                rs[k] = v.squeeze(dim=1)
        return rs


__all__ = ["ViTEncoder"]
