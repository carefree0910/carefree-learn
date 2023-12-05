import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from torch.nn import Module
from cftool.types import tensor_dict_type

from ..common import register_encoder
from ...core import ImgToPatches
from ...core import MixedStackedEncoder


@register_encoder("vit")
class ViTEncoder(Module):
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
        embedding_norm: Optional[Module] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        feedforward_kwargs: Optional[Dict[str, Any]] = None,
        use_head_token: bool = True,
        head_pooler: Optional[str] = "mean",
        use_positional_encoding: bool = True,
        norm_after_head: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            latent_dim=latent_dim,
            to_patches_type=to_patches_type,
            to_patches_config=to_patches_config,
        )
        # to patches
        self.to_patches = ImgToPatches.make(to_patches_type, to_patches_config)
        # transformer
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", latent_dim // 64)
        self.encoder = MixedStackedEncoder(
            latent_dim,
            self.to_patches.num_patches,
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
        )
        if output_dim is None:
            self.output_projection = None
        else:
            init = (latent_dim**-0.5) * torch.randn(latent_dim, output_dim)
            self.output_projection = torch.nn.Parameter(init)

    def forward(
        self,
        net: Tensor,
        *,
        hw: Optional[Tuple[int, int]] = None,
        hwp: Optional[Tuple[int, int, int]] = None,
        deterministic: bool = False,
    ) -> Tensor:
        patches, hw = self.to_patches(net, deterministic=deterministic)
        net = self.encoder(patches, hw=hw, hwp=hwp, deterministic=deterministic)
        if self.output_projection is not None:
            net = net @ self.output_projection
        return net


__all__ = [
    "ViTEncoder",
]
