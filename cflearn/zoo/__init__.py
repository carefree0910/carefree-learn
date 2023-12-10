from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from pathlib import Path
from torch.nn import Module

from .utils import *
from .common import *
from ..modules import generators


# translator


def esr(pretrained: bool = True) -> Module:
    return load_module("sr/esr", pretrained=pretrained)


def esr_anime(pretrained: bool = True) -> Module:
    return load_module("sr/esr.anime", pretrained=pretrained)


# clip


def clip(pretrained: bool = True, **kwargs: Any) -> Module:
    return load_module("multimodal/clip", pretrained=pretrained, **kwargs)


def chinese_clip(pretrained: bool = True, **kwargs: Any) -> Module:
    return load_module("multimodal/clip.chinese", pretrained=pretrained, **kwargs)


def open_clip_ViT_H_14(pretrained: bool = True, **kwargs: Any) -> Module:
    return load_module(
        "multimodal/clip.open_clip_ViT_H_14",
        pretrained=pretrained,
        **kwargs,
    )


# ae


def _ae(config: str, size: int, pretrained: bool, **kwargs: Any) -> Module:
    if pretrained and size != 256:
        raise ValueError(f"pretrained `{config}` should have `size`=256")
    kwargs["img_size"] = size
    kwargs["pretrained"] = pretrained
    return load_module(config, **kwargs)


def ae_kl_f4(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/kl.f4", size, pretrained, **kwargs)


def ae_kl_f8(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/kl.f8", size, pretrained, **kwargs)


def ae_kl_f16(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/kl.f16", size, pretrained, **kwargs)


def ae_vq_f4(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/vq.f4", size, pretrained, **kwargs)


def ae_vq_f4_no_attn(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/vq.f4_no_attn", size, pretrained, **kwargs)


def ae_vq_f8(size: int = 256, pretrained: bool = True, **kwargs: Any) -> Module:
    return _ae("ae/vq.f8", size, pretrained, **kwargs)


# sd


class SDVersions(str, Enum):
    v1_5_BC = ""
    v1_5 = "v1.5"
    ANIME = "anime"
    ANIME_ANYTHING = "anime_anything"
    ANIME_HYBRID = "anime_hybrid"
    ANIME_GUOFENG = "anime_guofeng"
    ANIME_ORANGE = "anime_orange"
    DREAMLIKE = "dreamlike_v1"


def get_sd_tag(version: Optional[str]) -> str:
    if version is None:
        version = SDVersions.v1_5
    if version == SDVersions.v1_5_BC:
        return "v1.5"
    if version == SDVersions.ANIME:
        return "anime_nai"
    if version == SDVersions.ANIME_ANYTHING:
        return "anime_anything_v3"
    if version == SDVersions.ANIME_HYBRID:
        return "anime_hybrid_v1"
    if version == SDVersions.ANIME_GUOFENG:
        return "anime_guofeng3"
    if version == SDVersions.ANIME_ORANGE:
        return "anime_orange2"
    return version


def _ldm(
    config: str,
    latent_size: int,
    latent_in_channels: int,
    latent_out_channels: int,
    *,
    pretrained: bool = False,
    tag: Optional[str] = None,
    download_root: Optional[Path] = None,
    download_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Module:
    kwargs["img_size"] = latent_size
    kwargs["in_channels"] = latent_in_channels
    kwargs["out_channels"] = latent_out_channels
    return load_module(
        config,
        pretrained=pretrained,
        tag=tag,
        download_root=download_root,
        download_kwargs=download_kwargs,
        prefix_module=generators,
        **kwargs,
    )


def ldm_sd(version: Optional[str] = None, pretrained: bool = True) -> Module:
    return _ldm(
        "diffusion/ldm.sd",
        64,
        4,
        4,
        pretrained=pretrained,
        tag=f"ldm_sd_{get_sd_tag(version)}",
    )


def ldm_sd_inpainting(pretrained: bool = True) -> Module:
    return _ldm(
        "diffusion/ldm.sd_inpainting",
        64,
        9,
        4,
        pretrained=pretrained,
        tag="ldm.sd_inpainting",
    )
