import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from .schema import Encoder1DFromPatches
from .schema import Encoder2DFromPatches
from ....constants import LATENT_KEY
from ....modules.blocks import MixedStackedEncoder


@Encoder1DFromPatches.register("vit")
class ViTEncoder(Encoder1DFromPatches):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 384,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        embedding_norm: Optional[torch.nn.Module] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        feedforward_kwargs: Optional[Dict[str, Any]] = None,
        use_head_token: bool = True,
        head_pooler: Optional[str] = "mean",
        use_positional_encoding: bool = True,
        norm_after_head: bool = False,
        output_dim: Optional[int] = None,
        aux_heads: Optional[List[str]] = None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", latent_dim // 64)
        self.aux_heads = aux_heads
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.num_patches,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            channel_mixing_config=feedforward_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            embedding_norm=embedding_norm,
            residual_after_norm=residual_after_norm,
            latent_dim_ratio=feedforward_dim_ratio,
            head_pooler=head_pooler,
            use_head_token=use_head_token,
            use_positional_encoding=use_positional_encoding,
            is_vision_positional_encoding=True,
            norm_after_head=norm_after_head,
            aux_heads=aux_heads,
        )
        if output_dim is None:
            self.output_projection = None
        else:
            init = (latent_dim**-0.5) * torch.randn(latent_dim, output_dim)
            self.output_projection = torch.nn.Parameter(init)

    def forward(self, net: Tensor, **kwargs: Any) -> tensor_dict_type:
        latent = super().forward(net, **kwargs)
        rs = {LATENT_KEY: latent}
        if self.aux_heads is None:
            if self.output_projection is not None:
                rs[LATENT_KEY] = latent @ self.output_projection
        else:
            for i, k in enumerate(self.aux_heads + [LATENT_KEY]):
                rs[k] = latent[:, i]
        return rs


@Encoder2DFromPatches.register("vit")
class ViTEncoder2D(Encoder2DFromPatches):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_channels: int = 384,
        to_patches_type: str = "vanilla",
        to_patches_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        embedding_norm: Optional[torch.nn.Module] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        feedforward_kwargs: Optional[Dict[str, Any]] = None,
        use_positional_encoding: bool = True,
        norm_after_head: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_channels=latent_channels,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", latent_channels // 64)
        self.encoder = MixedStackedEncoder(
            latent_channels,
            self.num_patches,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            channel_mixing_config=feedforward_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            embedding_norm=embedding_norm,
            residual_after_norm=residual_after_norm,
            latent_dim_ratio=feedforward_dim_ratio,
            head_pooler=None,
            use_head_token=False,
            use_positional_encoding=use_positional_encoding,
            is_vision_positional_encoding=True,
            norm_after_head=norm_after_head,
        )

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        latent = super().forward(net, **kwargs)
        resolution = int(round(math.sqrt(self.num_patches)))
        latent = latent.view(latent.shape[0], resolution, resolution, -1)
        latent = latent.permute(0, 3, 1, 2).contiguous()
        return latent


__all__ = [
    "ViTEncoder",
    "ViTEncoder2D",
]
