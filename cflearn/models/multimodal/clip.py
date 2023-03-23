import torch

from torch import nn
from torch import Tensor
from typing import Optional
from cftool.cv import to_rgb
from cftool.array import l2_normalize
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import CenterCrop
from torchvision.transforms import InterpolationMode

from .schema import IPerceptor
from ...constants import INPUT_KEY
from ...constants import LATENT_KEY
from ...modules.blocks import HijackLinear
from ..cv.encoder.transformer import ViTEncoder
from ..nlp.encoder.transformer import TeTEncoder


@IPerceptor.register("clip")
class CLIP(IPerceptor):
    vit: Optional[ViTEncoder]
    token_embedding: Optional[nn.Embedding]
    token_type_embedding: Optional[nn.Embedding]
    text_transformer: Optional[TeTEncoder]
    text_latent_dropout: Optional[nn.Dropout]
    text_projection: Optional[HijackLinear]

    def __init__(
        self,
        img_size: int = 224,
        latent_dim: int = 512,
        *,
        # vision
        use_vision: bool = True,
        in_channels: int = 3,
        vision_latent_dim: int = 768,
        vision_patch_size: int = 32,
        vision_num_heads: int = 12,
        vision_num_layers: int = 12,
        vision_norm_eps: float = 1.0e-5,
        vision_feedforward_activation: str = "quick_gelu",
        # text
        use_text: bool = True,
        vocab_size: int = 49408,
        context_length: int = 77,
        use_text_triu_attn_mask: bool = True,
        token_type_size: Optional[int] = None,
        text_latent_dim: int = 512,
        text_padding_idx: int = 0,
        use_text_embedding_norm: bool = False,
        text_embedding_dropout: Optional[bool] = None,
        text_dropout: float = 0.0,
        text_num_heads: int = 8,
        text_num_layers: int = 12,
        text_norm_position: str = "pre_norm",
        text_norm_eps: float = 1.0e-5,
        text_feedforward_activation: str = "quick_gelu",
        text_head_pooler: Optional[str] = None,
    ):
        super().__init__(img_size, context_length)
        # vision
        if not use_vision:
            self.vit = None
        else:
            self._init_vision(
                img_size,
                latent_dim,
                in_channels,
                vision_latent_dim,
                vision_patch_size,
                vision_num_heads,
                vision_num_layers,
                vision_norm_eps,
                vision_feedforward_activation,
            )
        # text
        if not use_text:
            self.token_embedding = None
            self.token_type_embedding = None
            self.text_transformer = None
            self.text_latent_dropout = None
            self.text_projection = None
        else:
            self._init_text(
                latent_dim,
                vocab_size,
                context_length,
                use_text_triu_attn_mask,
                token_type_size,
                text_latent_dim,
                text_padding_idx,
                use_text_embedding_norm,
                text_embedding_dropout,
                text_dropout,
                text_num_heads,
                text_num_layers,
                text_norm_position,
                text_norm_eps,
                text_feedforward_activation,
                text_head_pooler,
            )
        # initialize
        self.reset_parameters()

    def _init_vision(
        self,
        img_size: int,
        latent_dim: int,
        in_channels: int,
        vision_latent_dim: int,
        vision_patch_size: int,
        vision_num_heads: int,
        vision_num_layers: int,
        vision_norm_eps: float,
        vision_feedforward_activation: str,
    ) -> None:
        self.vision_latent_dim = vision_latent_dim
        self.vit = ViTEncoder(
            img_size=img_size,
            patch_size=vision_patch_size,
            in_channels=in_channels,
            latent_dim=self.vision_latent_dim,
            to_patches_config={"bias": False},
            num_layers=vision_num_layers,
            norm_kwargs={"eps": vision_norm_eps},
            embedding_norm=nn.LayerNorm(self.vision_latent_dim, vision_norm_eps),
            attention_kwargs={"num_heads": vision_num_heads},
            feedforward_kwargs={"activation": vision_feedforward_activation},
            norm_after_head=True,
            output_dim=latent_dim,
        )

    def _init_text(
        self,
        latent_dim: int,
        vocab_size: int,
        context_length: int,
        use_text_triu_attn_mask: bool,
        token_type_size: Optional[int],
        text_latent_dim: int,
        text_padding_idx: int,
        use_text_embedding_norm: bool,
        text_embedding_dropout: Optional[bool],
        text_dropout: float,
        text_num_heads: int,
        text_num_layers: int,
        text_norm_position: str,
        text_norm_eps: float,
        text_feedforward_activation: str,
        text_head_pooler: Optional[str],
    ) -> None:
        self.text_num_layers = text_num_layers
        self.text_latent_dim = text_latent_dim
        self.token_embedding = nn.Embedding(
            vocab_size,
            text_latent_dim,
            padding_idx=text_padding_idx,
        )
        if token_type_size is None:
            self.token_type_embedding = None
        else:
            self.token_type_embedding = nn.Embedding(token_type_size, text_latent_dim)
        if not use_text_embedding_norm:
            text_embedding_norm = None
        else:
            text_embedding_norm = nn.LayerNorm(text_latent_dim, text_norm_eps)
        self.text_transformer = TeTEncoder(
            text_latent_dim,
            context_length,
            use_triu_attn_mask=use_text_triu_attn_mask,
            num_layers=text_num_layers,
            dropout=text_dropout,
            norm_position=text_norm_position,
            norm_kwargs={"eps": text_norm_eps},
            embedding_norm=text_embedding_norm,
            embedding_dropout=text_embedding_dropout,
            attention_kwargs={"num_heads": text_num_heads},
            feedforward_kwargs={"activation": text_feedforward_activation},
            head_pooler=text_head_pooler,
        )
        self.text_head_pooler = text_head_pooler
        self.text_latent_dropout = nn.Dropout(text_dropout)
        self.text_projection = HijackLinear(text_latent_dim, latent_dim)

    def reset_parameters(self) -> None:
        if self.token_embedding is not None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
        tld = self.text_latent_dim
        if self.text_transformer is not None:
            text_encoder = self.text_transformer.encoder
            nn.init.normal_(text_encoder.pos_encoding.pos_encoding, std=0.01)
            proj_std = (tld**-0.5) * ((2 * self.text_num_layers) ** -0.5)
            attn_std = tld**-0.5
            fc_std = (2 * tld) ** -0.5
            for block in text_encoder.mixing_blocks:
                attn = block.token_mixing.net
                mlp = block.channel_mixing.net
                nn.init.normal_(attn.in_w, std=attn_std)
                nn.init.normal_(attn.out_linear.weight, std=proj_std)
                nn.init.normal_(mlp[0].weight, std=fc_std)
                nn.init.normal_(mlp[3].weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=tld**-0.5)
            nn.init.zeros_(self.text_projection.bias)

    def encode_image(self, image: Tensor) -> Tensor:
        if self.vit is None:
            raise ValueError(
                "`vit` is not initialized, "
                "please set `use_vision=True` when initializing `CLIP`"
            )
        net = self.vit(image, determinate=True)[LATENT_KEY]
        return l2_normalize(net)

    def encode_text(
        self,
        indices: Tensor,
        *,
        apply_pooling: bool = True,
        determinate: bool = True,
        clip_skip: int = 0,
    ) -> Tensor:
        fmt = (
            "`{}` is not initialized, "
            "please set `use_text=True` when initializing `CLIP`"
        )
        if self.token_embedding is None:
            raise ValueError(fmt.format("token_embedding"))
        if self.text_transformer is None:
            raise ValueError(fmt.format("text_transformer"))
        if self.text_latent_dropout is None:
            raise ValueError(fmt.format("text_latent_dropout"))
        if self.text_projection is None:
            raise ValueError(fmt.format("text_projection"))
        net = self.token_embedding(indices)
        if self.token_type_embedding is not None:
            token_type = torch.zeros_like(indices)
            net = net + self.token_type_embedding(token_type)
        kw = dict(clip_skip=clip_skip, determinate=determinate)
        net = self.text_transformer(0, {INPUT_KEY: net}, **kw)[LATENT_KEY]
        if not apply_pooling:
            return net
        # 'pool' the latent net if pooler is not provided
        if self.text_head_pooler is None:
            batch_size = net.shape[0]
            if self.training or batch_size > 1:
                net = net[torch.arange(batch_size), indices.argmax(dim=-1)]
            else:
                # ONNX compatibility
                net = net[:, torch.nonzero(indices)[:, 1][-1]]
        net = self.text_latent_dropout(net)
        net = self.text_projection(net)
        return l2_normalize(net)

    def get_transform(self) -> Compose:
        return Compose(
            [
                Resize(self.img_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self.img_size),
                to_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


__all__ = ["CLIP"]
