from typing import Any
from typing import Optional

from ..protocol import ModelProtocol
from .zoo.core import DLZoo
from .internal_.pipeline import DLPipeline
from ..misc.toolkit import download_model


# cv

# clf


def cct_large(img_size: int, num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("clf/cct.large", **kwargs)


def cct_large_model(img_size: int, num_classes: int, **kwargs: Any) -> ModelProtocol:
    kwargs["img_size"] = img_size
    kwargs["num_classes"] = num_classes
    return DLZoo.load_model("clf/cct.large", **kwargs)


def cct_large_224(
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    if pretrained:
        model_config = kwargs.setdefault("model_config", {})
        model_config["encoder1d_pretrained_name"] = "cct_large_224"
    return DLZoo.load_pipeline("clf/cct.large_224", **kwargs)


def cct_large_224_model(
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> ModelProtocol:
    kwargs["num_classes"] = num_classes
    if pretrained:
        model_config = kwargs.setdefault("model_config", {})
        model_config["encoder1d_pretrained_name"] = "cct_large_224"
    return DLZoo.load_model("clf/cct.large_224", **kwargs)


def cct_large_384(
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    if pretrained:
        model_config = kwargs.setdefault("model_config", {})
        model_config["encoder1d_pretrained_name"] = "cct_large_384"
    return DLZoo.load_pipeline("clf/cct.large_384", **kwargs)


def cct_large_384_model(
    num_classes: int,
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> ModelProtocol:
    kwargs["num_classes"] = num_classes
    if pretrained:
        model_config = kwargs.setdefault("model_config", {})
        model_config["encoder1d_pretrained_name"] = "cct_large_384"
    return DLZoo.load_model("clf/cct.large_384", **kwargs)


def resnet18(num_classes: int, pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_pipeline("clf/resnet18", **kwargs)


def resnet18_gray(num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("clf/resnet18.gray", **kwargs)


def resnet101(num_classes: int, pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    model_config = kwargs.setdefault("model_config", {})
    encoder1d_config = model_config.setdefault("encoder1d_config", {})
    encoder1d_config["pretrained"] = pretrained
    return DLZoo.load_pipeline("clf/resnet101", **kwargs)


# gan


def vanilla_gan(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/vanilla", **kwargs)


def vanilla_gan_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/vanilla.gray", **kwargs)


def siren_gan(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/siren", **kwargs)


def siren_gan_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("gan/siren.gray", **kwargs)


# generator


def pixel_cnn(num_classes: int, **kwargs: Any) -> DLPipeline:
    kwargs["num_classes"] = num_classes
    return DLZoo.load_pipeline("generator/pixel_cnn", **kwargs)


# multimodal


def clip_vqgan_aligner(**kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("multimodal/clip_vqgan_aligner", **kwargs)


# segmentor


def u2net(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_model(pretrained: bool = False, **kwargs: Any) -> ModelProtocol:
    return DLZoo.load_model("segmentor/u2net", pretrained=pretrained, **kwargs)


def u2net_lite(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_lite_model(pretrained: bool = False, **kwargs: Any) -> ModelProtocol:
    return DLZoo.load_model("segmentor/u2net.lite", pretrained=pretrained, **kwargs)


def u2net_finetune(ckpt: Optional[str] = None, **kwargs: Any) -> DLPipeline:
    if ckpt is None:
        kwargs["pretrained_ckpt"] = download_model("u2net")
    return DLZoo.load_pipeline("segmentor/u2net.finetune", **kwargs)


def u2net_lite_finetune(ckpt: Optional[str] = None, **kwargs: Any) -> DLPipeline:
    if ckpt is None:
        kwargs["pretrained_ckpt"] = download_model("u2net.lite")
    return DLZoo.load_pipeline("segmentor/u2net.finetune_lite", **kwargs)


def u2net_refine(lv1_model_ckpt_path: str, **kwargs: Any) -> DLPipeline:
    kwargs["lv1_model_ckpt_path"] = lv1_model_ckpt_path
    return DLZoo.load_pipeline("segmentor/u2net.refine", **kwargs)


def u2net_lite_refine(lv1_model_ckpt_path: str, **kwargs: Any) -> DLPipeline:
    kwargs["lv1_model_ckpt_path"] = lv1_model_ckpt_path
    return DLZoo.load_pipeline("segmentor/u2net.refine_lite", **kwargs)


# ssl


def dino(**kwargs: Any) -> DLPipeline:
    return DLZoo.load_pipeline("ssl/dino", **kwargs)


# style transfer


def adain(pretrained: bool = False, **kwargs: Any) -> DLPipeline:
    kwargs["pretrained"] = pretrained
    return DLZoo.load_pipeline("style_transfer/adain", **kwargs)


# vae


def vanilla_vae(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla", **kwargs)


def vanilla_vae_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vanilla.gray", **kwargs)


def siren_vae(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/siren", **kwargs)


def siren_vae_gray(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/siren.gray", **kwargs)


def vq_vae_lite(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vq.lite", **kwargs)


def vq_vae_gray_lite(img_size: int, **kwargs: Any) -> DLPipeline:
    kwargs["img_size"] = img_size
    return DLZoo.load_pipeline("vae/vq.gray_lite", **kwargs)


__all__ = [
    "cct_large",
    "cct_large_model",
    "cct_large_224",
    "cct_large_224_model",
    "cct_large_384",
    "cct_large_384_model",
    "resnet18",
    "resnet18_gray",
    "resnet101",
    "vanilla_gan",
    "vanilla_gan_gray",
    "siren_gan",
    "siren_gan_gray",
    "pixel_cnn",
    "clip_vqgan_aligner",
    "u2net",
    "u2net_model",
    "u2net_lite",
    "u2net_lite_model",
    "u2net_finetune",
    "u2net_lite_finetune",
    "u2net_refine",
    "u2net_lite_refine",
    "dino",
    "adain",
    "vanilla_vae",
    "vanilla_vae_gray",
    "siren_vae",
    "siren_vae_gray",
    "vq_vae_lite",
    "vq_vae_gray_lite",
]
