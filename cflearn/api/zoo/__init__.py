from typing import Any
from typing import Union
from cftool.types import tensor_dict_type

from ...schema import DLConfig
from ...zoo.core import DLZoo
from ...zoo.core import TPipeline
from ...models.cv.diffusion.utils import CONCAT_TYPE


TLoadZoo = Union[TPipeline, DLConfig]


def _load(model: str, **kwargs: Any) -> TLoadZoo:
    fn = DLZoo.load_config if kwargs.pop("load_config", False) else DLZoo.load_pipeline
    return fn(model, **kwargs)  # type: ignore


# ae


def ae_kl_f4(size: int = 256, pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    if pretrained and size != 256:
        raise ValueError("pretrained `ae_kl_f4` should have `size`=256")
    kwargs["img_size"] = size
    kwargs["pretrained"] = pretrained
    return _load("ae/kl.f4", **kwargs)


def ae_kl_f8(size: int = 256, pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    if pretrained and size != 256:
        raise ValueError("pretrained `ae_kl_f8` should have `size`=256")
    kwargs["img_size"] = size
    kwargs["pretrained"] = pretrained
    return _load("ae/kl.f8", **kwargs)


def ae_kl_f16(size: int = 256, pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    if pretrained and size != 256:
        raise ValueError("pretrained `ae_kl_f16` should have `size`=256")
    kwargs["img_size"] = size
    kwargs["pretrained"] = pretrained
    return _load("ae/kl.f16", **kwargs)


# diffusion


def ddpm(img_size: int = 256, **kwargs: Any) -> TLoadZoo:
    kwargs["img_size"] = img_size
    return _load("diffusion/ddpm", **kwargs)


def _ldm(
    model: str,
    latent_size: int,
    latent_in_channels: int,
    latent_out_channels: int,
    **kwargs: Any,
) -> TLoadZoo:
    kwargs["img_size"] = latent_size
    kwargs["in_channels"] = latent_in_channels
    kwargs["out_channels"] = latent_out_channels
    model_config = kwargs.setdefault("model_config", {})
    first_stage_kw = model_config.setdefault("first_stage_config", {})
    first_stage_kw.setdefault("pretrained", False)
    first_stage_model_config = first_stage_kw.setdefault("model_config", {})
    use_loss = first_stage_model_config.setdefault("use_loss", False)
    if not use_loss:

        def state_callback(states: tensor_dict_type) -> tensor_dict_type:
            for key in list(states.keys()):
                if key.startswith("loss"):
                    states.pop(key)
            return states

        first_stage_kw["states_callback"] = state_callback
    return _load(model, **kwargs)


def ldm(
    latent_size: int = 32,
    latent_in_channels: int = 4,
    latent_out_channels: int = 4,
    **kwargs: Any,
) -> TLoadZoo:
    return _ldm(
        "diffusion/ldm",
        latent_size,
        latent_in_channels,
        latent_out_channels,
        **kwargs,
    )


def ldm_vq(
    latent_size: int = 64,
    latent_in_channels: int = 3,
    latent_out_channels: int = 3,
    **kwargs: Any,
) -> TLoadZoo:
    return _ldm(
        "diffusion/ldm.vq",
        latent_size,
        latent_in_channels,
        latent_out_channels,
        **kwargs,
    )


def ldm_sd(pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    return _ldm("diffusion/ldm.sd", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_sd_tag(tag: str, pretrained: bool = True) -> TLoadZoo:
    return ldm_sd(pretrained, download_name=f"ldm_sd_{tag}")


def ldm_sd_inpainting(pretrained: bool = True, **kw: Any) -> TLoadZoo:
    return _ldm("diffusion/ldm.sd_inpainting", 64, 9, 4, pretrained=pretrained, **kw)


def ldm_sd_v2(pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    return _ldm("diffusion/ldm.sd_v2", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_sd_v2_base(pretrained: bool = True, **kwargs: Any) -> TLoadZoo:
    return _ldm("diffusion/ldm.sd_v2_base", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_celeba_hq(pretrained: bool = True) -> TLoadZoo:
    return ldm_vq(
        pretrained=pretrained,
        download_name="ldm_celeba_hq",
        model_config=dict(
            ema_decay=None,
            first_stage_config=dict(
                pretrained=False,
            ),
        ),
    )


def ldm_inpainting(pretrained: bool = True) -> TLoadZoo:
    return ldm_vq(
        pretrained=pretrained,
        latent_in_channels=7,
        download_name="ldm_inpainting",
        model_config=dict(
            ema_decay=None,
            start_channels=256,
            num_heads=8,
            num_head_channels=None,
            resample_with_resblock=True,
            condition_type=CONCAT_TYPE,
            first_stage_config=dict(
                pretrained=False,
                model_config=dict(
                    attention_type="none",
                ),
            ),
        ),
    )


def ldm_sr(pretrained: bool = True) -> TLoadZoo:
    return ldm_vq(
        pretrained=pretrained,
        latent_in_channels=6,
        download_name="ldm_sr",
        model_config=dict(
            ema_decay=None,
            start_channels=160,
            attention_downsample_rates=[8, 16],
            channel_multipliers=[1, 2, 2, 4],
            condition_type=CONCAT_TYPE,
            first_stage_config=dict(
                pretrained=False,
            ),
        ),
    )


def ldm_semantic(pretrained: bool = True) -> TLoadZoo:
    return ldm_vq(
        pretrained=pretrained,
        latent_size=128,
        latent_in_channels=6,
        download_name="ldm_semantic",
        model_config=dict(
            ema_decay=None,
            start_channels=128,
            num_heads=8,
            num_head_channels=None,
            attention_downsample_rates=[8, 16, 32],
            channel_multipliers=[1, 4, 8],
            condition_type=CONCAT_TYPE,
            condition_model="rescaler",
            condition_config=dict(
                num_stages=2,
                in_channels=182,
                out_channels=3,
            ),
            first_stage_config=dict(
                pretrained=False,
            ),
        ),
    )
