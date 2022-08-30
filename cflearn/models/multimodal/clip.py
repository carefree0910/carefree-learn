import torch

from torch import nn
from torch import Tensor
from typing import Optional
from cftool.misc import shallow_copy_dict
from cftool.array import l2_normalize
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import CenterCrop
from torchvision.transforms import InterpolationMode

from .protocol import IPerceptor
from ...constants import INPUT_KEY
from ...constants import LATENT_KEY
from ..cv.encoder.transformer import ViTEncoder
from ..nlp.encoder.transformer import TeTEncoder

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


@IPerceptor.register("clip")
class CLIP(IPerceptor):
    def __init__(
        self,
        img_size: int = 224,
        latent_dim: int = 512,
        *,
        # vision
        in_channels: int = 3,
        vision_latent_dim: int = 768,
        vision_patch_size: int = 32,
        vision_num_heads: int = 12,
        vision_num_layers: int = 12,
        vision_norm_eps: float = 1.0e-5,
        # text
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
        self.vision_latent_dim = vision_latent_dim
        feedforward_kwargs = {"activation": "quick_gelu"}
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
            feedforward_kwargs=feedforward_kwargs,
            norm_after_head=True,
            output_dim=latent_dim,
        )
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
        feedforward_kwargs = shallow_copy_dict(feedforward_kwargs)
        feedforward_kwargs["activation"] = text_feedforward_activation
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
            feedforward_kwargs=feedforward_kwargs,
            head_pooler=text_head_pooler,
        )
        self.text_head_pooler = text_head_pooler
        self.text_latent_dropout = nn.Dropout(text_dropout)
        self.text_projection = nn.Linear(text_latent_dim, latent_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        text_encoder = self.text_transformer.encoder
        nn.init.normal_(text_encoder.pos_encoding.pos_encoding, std=0.01)
        proj_std = (self.text_latent_dim**-0.5) * ((2 * self.text_num_layers) ** -0.5)
        attn_std = self.text_latent_dim**-0.5
        fc_std = (2 * self.text_latent_dim) ** -0.5
        for block in text_encoder.mixing_blocks:
            attn = block.token_mixing.net
            mlp = block.channel_mixing.net
            nn.init.normal_(attn.in_w, std=attn_std)
            nn.init.normal_(attn.out_linear.weight, std=proj_std)
            nn.init.normal_(mlp[0].weight, std=fc_std)
            nn.init.normal_(mlp[3].weight, std=proj_std)
        nn.init.normal_(self.text_projection.weight, std=self.text_latent_dim**-0.5)
        nn.init.zeros_(self.text_projection.bias)

    def encode_image(self, image: Tensor) -> Tensor:
        net = self.vit(image, determinate=True)[LATENT_KEY]
        return l2_normalize(net)

    def encode_text(self, text: Tensor) -> Tensor:
        net = self.token_embedding(text)
        if self.token_type_embedding is not None:
            token_type = torch.zeros_like(text)
            net = net + self.token_type_embedding(token_type)
        net = self.text_transformer(0, {INPUT_KEY: net}, determinate=True)[LATENT_KEY]
        # 'pool' the latent net if pooler is not provided
        if self.text_head_pooler is None:
            batch_size = net.shape[0]
            if self.training or batch_size > 1:
                net = net[torch.arange(batch_size), text.argmax(dim=-1)]
            else:
                # ONNX compatibility
                net = net[:, torch.nonzero(text)[:, 1][-1]]
        net = self.text_latent_dropout(net)
        net = self.text_projection(net)
        return l2_normalize(net)

    def get_transform(self) -> Compose:
        if to_rgb is None:
            raise ValueError("`carefree-cv` is needed to use `get_transform`")
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
