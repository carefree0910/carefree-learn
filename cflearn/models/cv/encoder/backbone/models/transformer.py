import torch

from torch import nn
from typing import Any
from typing import List
from typing import Optional
from cftool.misc import shallow_copy_dict

from ..register import register_backbone
from ......modules.blocks import ImgToPatches
from ......modules.blocks import MixedStackedEncoder


class MixViTBlock(nn.Module):
    def __init__(self, embed: ImgToPatches, encoder: MixedStackedEncoder):
        super().__init__()
        self.embed = embed
        self.encoder = encoder

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        net, hw = self.embed(net)
        net = self.encoder(net, hw)
        net = net.view(net.shape[0], *hw, -1).permute(0, 3, 1, 2).contiguous()
        return net


class MixViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dims: List[int],
        *,
        num_heads_list: List[int],
        feedforward_dim_ratios: List[float],
        num_layers_list: List[int],
        reduction_ratios: List[int],
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_type: Optional[str] = "layer",
        **attention_kwargs: Any,
    ):
        super().__init__()
        # patch embeddings
        patch_embeds = []
        patch_in_channels = in_channels
        for i, (patch_size, stride) in enumerate(zip([7, 3, 3, 3], [4, 2, 2, 2])):
            latent_dim = latent_dims[i]
            to_patches_config = {
                "img_size": 0,
                "patch_size": patch_size,
                "in_channels": patch_in_channels,
                "latent_dim": latent_dim,
                "stride": stride,
            }
            patch_embeds.append(ImgToPatches.make("overlap", to_patches_config))
            patch_in_channels = latent_dim
        # transformer encoders
        total_num_layers = sum(num_layers_list)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_num_layers)]
        cursor = 0
        encoders = []
        attention_kwargs.setdefault("bias", True)
        for i, (latent_dim, num_heads) in enumerate(zip(latent_dims, num_heads_list)):
            ff_ratio = feedforward_dim_ratios[i]
            num_layers = num_layers_list[i]
            ak = shallow_copy_dict(attention_kwargs)
            ak["num_heads"] = num_heads
            ak["reduction_ratio"] = reduction_ratios[i]
            encoders.append(
                MixedStackedEncoder(
                    latent_dim,
                    0,
                    token_mixing_type="attention",
                    token_mixing_config=ak,
                    channel_mixing_type="mix_ff",
                    num_layers=num_layers,
                    dropout=dropout,
                    dpr_list=dpr[cursor : cursor + num_layers],
                    norm_type=norm_type,
                    latent_dim_ratio=ff_ratio,
                    use_head_token=False,
                    head_pooler=None,
                    use_positional_encoding=False,
                )
            )
            cursor += num_layers
        # stages
        for i, (patch_embed, encoder) in enumerate(zip(patch_embeds, encoders)):
            setattr(self, f"stage{i + 1}", MixViTBlock(patch_embed, encoder))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        net = self.stage1(net)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        return net


@register_backbone(
    "mix_vit",
    [64, 128, 320, 512],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4="stage4",
    ),
)
def mix_vit(pretrained: bool = False, *, in_channels: int = 3) -> MixViT:
    if pretrained:
        raise ValueError("`MixViT` does not support `pretrained`")
    return MixViT(
        in_channels,
        [64, 128, 320, 512],
        num_heads_list=[1, 2, 5, 8],
        feedforward_dim_ratios=[4.0] * 4,
        num_layers_list=[3, 4, 18, 3],
        reduction_ratios=[8, 4, 2, 1],
    )


@register_backbone(
    "mix_vit_lite",
    [32, 64, 160, 256],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4="stage4",
    ),
)
def mix_vit_lite(pretrained: bool = False, *, in_channels: int = 3) -> MixViT:
    if pretrained:
        raise ValueError("`MixViT` does not support `pretrained`")
    return MixViT(
        in_channels,
        [32, 64, 160, 256],
        num_heads_list=[1, 2, 5, 8],
        feedforward_dim_ratios=[4.0] * 4,
        num_layers_list=[2] * 4,
        reduction_ratios=[8, 4, 2, 1],
    )


@register_backbone(
    "mix_vit_large",
    [64, 128, 320, 512],
    dict(
        stage1="stage1",
        stage2="stage2",
        stage3="stage3",
        stage4="stage4",
    ),
)
def mix_vit_large(pretrained: bool = False, *, in_channels: int = 3) -> MixViT:
    if pretrained:
        raise ValueError("`MixViT` does not support `pretrained`")
    return MixViT(
        in_channels,
        [64, 128, 320, 512],
        num_heads_list=[1, 2, 5, 8],
        feedforward_dim_ratios=[4.0] * 4,
        num_layers_list=[3, 6, 40, 3],
        reduction_ratios=[8, 4, 2, 1],
    )


__all__ = [
    "mix_vit",
    "mix_vit_lite",
    "mix_vit_large",
]
