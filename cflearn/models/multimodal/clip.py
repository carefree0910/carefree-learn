import torch

from PIL import Image
from torch import nn
from torch import Tensor
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import CenterCrop
from torchvision.transforms import InterpolationMode

from .protocol import PerceptorProtocol
from ...constants import INPUT_KEY
from ...constants import LATENT_KEY
from ...misc.toolkit import l2_normalize
from ..cv.encoder.transformer import ViTEncoder
from ..nlp.encoder.transformer import TeTEncoder


def to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


@PerceptorProtocol.register("clip")
class CLIP(PerceptorProtocol):
    def __init__(
        self,
        img_size: int = 224,
        latent_dim: int = 512,
        *,
        # vision
        in_channels: int = 3,
        vision_latent_expand: float = 1.5,
        vision_patch_size: int = 32,
        vision_num_layers: int = 12,
        vision_num_heads: int = 12,
        # text
        vocab_size: int = 49408,
        context_length: int = 77,
        text_latent_expand: float = 1.0,
        text_num_layers: int = 12,
        text_num_heads: int = 8,
    ):
        super().__init__(img_size, context_length)
        self.vision_latent_dim = int(round(latent_dim * vision_latent_expand))
        feedforward_kwargs = {"activation": "quick_gelu"}
        self.vit = ViTEncoder(
            img_size,
            patch_size=vision_patch_size,
            in_channels=in_channels,
            latent_dim=self.vision_latent_dim,
            to_patches_config={"bias": False},
            num_layers=vision_num_layers,
            norm_kwargs={"eps": 1.0e-5},
            first_norm=nn.LayerNorm(self.vision_latent_dim),
            attention_kwargs={"num_heads": vision_num_heads},
            feedforward_kwargs=feedforward_kwargs,
            norm_after_head=True,
            output_dim=latent_dim,
        )
        self.text_num_layers = text_num_layers
        self.text_latent_dim = int(round(latent_dim * text_latent_expand))
        self.token_embedding = nn.Embedding(vocab_size, self.text_latent_dim)
        self.text_transformer = TeTEncoder(
            self.text_latent_dim,
            use_triu_attn_mask=True,
            num_layers=text_num_layers,
            norm_kwargs={"eps": 1.0e-5},
            attention_kwargs={"num_heads": text_num_heads},
            feedforward_kwargs=feedforward_kwargs,
        )
        projection_shape = self.text_latent_dim, latent_dim
        self.text_projection = nn.Parameter(torch.empty(*projection_shape))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        text_encoder = self.text_transformer.encoder
        nn.init.normal_(text_encoder.pos_encoding.pos_encoding, std=0.01)
        proj_std = (self.text_latent_dim ** -0.5) * ((2 * self.text_num_layers) ** -0.5)
        attn_std = self.text_latent_dim ** -0.5
        fc_std = (2 * self.text_latent_dim) ** -0.5
        for block in text_encoder.mixing_blocks:
            attn = block.token_mixing.module.net
            mlp = block.channel_mixing.module.net
            nn.init.normal_(attn.in_w, std=attn_std)
            nn.init.normal_(attn.out_linear.weight, std=proj_std)
            nn.init.normal_(mlp[0].weight, std=fc_std)
            nn.init.normal_(mlp[3].weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=self.text_latent_dim ** -0.5)

    def encode_image(self, image: Tensor) -> Tensor:
        net = self.vit(0, {INPUT_KEY: image}, determinate=True)[LATENT_KEY]
        return l2_normalize(net)

    def encode_text(self, text: Tensor) -> Tensor:
        net = self.token_embedding(text)
        net = self.text_transformer(0, {INPUT_KEY: net}, determinate=True)[LATENT_KEY]
        batch_size = net.shape[0]
        if self.training or batch_size > 1:
            net = net[torch.arange(batch_size), text.argmax(dim=-1)]
        else:
            # ONNX compatibility
            net = net[:, torch.nonzero(text)[:, 1][-1]]
        net = net @ self.text_projection
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
