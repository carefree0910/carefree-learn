import json
import time
import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from PIL import ImageOps
from enum import Enum
from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import ContextManager
from pathlib import Path
from filelock import FileLock
from dataclasses import dataclass
from cftool.cv import to_rgb
from cftool.cv import to_uint8
from cftool.cv import read_image
from cftool.cv import save_images
from cftool.cv import restrict_wh
from cftool.cv import to_alpha_channel
from cftool.cv import get_suitable_size
from cftool.cv import ImageBox
from cftool.cv import ImageProcessor
from cftool.cv import ReadImageResponse
from cftool.misc import safe_execute
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.types import TNumberPair
from cftool.types import arr_type
from cftool.types import tensor_dict_type

from ..common import IAPI
from ..common import WeightsPool
from ..cv.annotator import annotators
from ..cv.annotator import Annotator
from ...zoo import ldm_sd
from ...zoo import get_sd_tag
from ...zoo import ldm_sd_inpainting
from ...data import ArrayData
from ...data import TensorBatcher
from ...schema import device_type
from ...schema import DataConfig
from ...modules import LDM
from ...modules import DDPM
from ...modules import Conv2d
from ...modules import DecoderInputs
from ...modules import StableDiffusion
from ...modules import DDIMMixin
from ...modules import DPMSolver
from ...modules import IKSampler
from ...toolkit import _get_file_size
from ...toolkit import slerp
from ...toolkit import freeze
from ...toolkit import new_seed
from ...toolkit import load_file
from ...toolkit import seed_everything
from ...toolkit import download_checkpoint
from ...toolkit import eval_context
from ...constants import INPUT_KEY
from ...parameters import OPT
from ...modules.core import walk_spatial_transformer_hooks
from ...modules.core import SpatialTransformerHooks
from ...modules.multimodal.diffusion.unet import ControlNet
from ...modules.multimodal.diffusion.utils import CONCAT_KEY
from ...modules.multimodal.diffusion.utils import CONCAT_TYPE
from ...modules.multimodal.diffusion.utils import HYBRID_TYPE
from ...modules.multimodal.diffusion.utils import CROSS_ATTN_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_END_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_START_KEY
from ...modules.multimodal.diffusion.utils import get_timesteps
from ...modules.multimodal.diffusion.cond_models import CLIPTextConditionModel

try:
    import cv2

    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
except:
    cv2 = INTER_LANCZOS4 = None


T = TypeVar("T", bound="DiffusionAPI")


def convert_external(
    m: "DiffusionAPI",
    tag: str,
    sub_folder: Optional[str],
    *,
    convert_fn: Optional[Callable] = None,
) -> Path:
    external_root = OPT.external_dir
    if sub_folder is not None:
        external_root = external_root / sub_folder
    lock_path = external_root / "load_external.lock"
    lock = FileLock(lock_path)
    with lock:
        converted_sizes_path = external_root / "sizes.json"
        sizes: Dict[str, int]
        if not converted_sizes_path.is_file():
            sizes = {}
        else:
            with converted_sizes_path.open("r") as f:
                sizes = json.load(f)
    converted_path = external_root / f"{tag}_converted.pt"
    if not converted_path.is_file():
        f_size = None
    else:
        f_size = _get_file_size(converted_path)
    v_size = sizes.get(tag)
    if f_size is None or v_size != f_size:
        if f_size is not None:
            print(f"> '{tag}' has been converted but size mismatch")
        print(f"> converting external weights '{tag}'")
        model_path = external_root / f"{tag}.ckpt"
        if not model_path.is_file():
            model_path = external_root / f"{tag}.pth"
        if not model_path.is_file():
            st_path = external_root / f"{tag}.safetensors"
            if not st_path.is_file():
                raise FileNotFoundError(f"cannot find '{tag}'")
            torch.save(load_file(st_path), model_path)
        if convert_fn is not None:
            d = convert_fn(model_path, m)
        else:
            import cflearn

            d = cflearn.scripts.sd.convert(model_path, m, load=False)
        torch.save(d, converted_path)
        sizes[tag] = _get_file_size(converted_path)
        with lock:
            with converted_sizes_path.open("w") as f:
                json.dump(sizes, f)
    return converted_path


def get_highres_steps(num_steps: int, fidelity: float) -> int:
    return int(num_steps / min(1.0 - fidelity, 0.999))


def get_size(
    size: Optional[Tuple[int, int]],
    anchor: int,
    max_wh: int,
) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    new_size = restrict_wh(*size, max_wh)
    return tuple(map(get_suitable_size, new_size, (anchor, anchor)))  # type: ignore


def get_txt_cond(t: Union[str, List[str]], n: Optional[int]) -> Tuple[List[str], int]:
    if n is None:
        n = 1 if isinstance(t, str) else len(t)
    if isinstance(t, str):
        t = [t] * n
    if len(t) != n:
        raise ValueError(
            f"`num_samples` ({n}) should be identical with "
            f"the number of `txt` ({len(t)})"
        )
    return t, n


def resize(
    inp: np.ndarray,
    wh: Tuple[int, int],
    interpolation: int = INTER_LANCZOS4,
) -> np.ndarray:
    return cv2.resize(inp, wh, interpolation=interpolation)


def adjust_lt_rb(lt_rb: ImageBox, w: int, h: int, padding: TNumberPair) -> ImageBox:
    l, t, r, b = lt_rb.tuple
    if padding is not None:
        if isinstance(padding, int):
            padding = padding, padding
        l = max(0, l - padding[0])
        t = max(0, t - padding[1])
        r = min(w, r + padding[0])
        b = min(h, b + padding[1])
    cropped_h, cropped_w = b - t, r - l
    # adjust lt_rb to make the cropped aspect ratio equals to the original one
    if cropped_h / cropped_w > h / w:
        dw = (int(cropped_h * w / h) - cropped_w) // 2
        dh = 0
    else:
        dw = 0
        dh = (int(cropped_w * h / w) - cropped_h) // 2
    if dw > 0:
        if l < dw:
            l = 0
            r = min(w, cropped_w + dw * 2)
        elif r + dw > w:
            r = w
            l = max(0, w - cropped_w - dw * 2)
        else:
            l -= dw
            r += dw
    if dh > 0:
        if t < dh:
            t = 0
            b = min(h, cropped_h + dh * 2)
        elif b + dh > h:
            b = h
            t = max(0, h - cropped_h - dh * 2)
        else:
            t -= dh
            b += dh
    return ImageBox(l, t, r, b)


def crop_masked_area(
    image_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    settings: "InpaintingSettings",
) -> "CropResponse":
    image = image_tensor[0].transpose(1, 2, 0)
    mask = mask_tensor[0, 0]
    h, w = image.shape[:2]
    lt_rb = ImageBox.from_mask(to_uint8(mask), settings.mask_binary_threshold)
    lt_rb = adjust_lt_rb(lt_rb, w, h, settings.mask_padding)
    # finalize
    if settings.target_wh is not None:
        if isinstance(settings.target_wh, int):
            w = h = settings.target_wh
        else:
            w, h = settings.target_wh
    cropped_image = lt_rb.crop(image)
    cropped_mask = lt_rb.crop(mask)
    resized_image = resize(cropped_image, (w, h))
    resized_mask = resize(cropped_mask, (w, h), cv2.INTER_NEAREST)
    resized_image = resized_image.transpose(2, 0, 1)[None]
    resized_mask = resized_mask[None, None]
    return CropResponse(lt_rb, (w, h), cropped_mask, resized_image, resized_mask)


def normalize_image_to_diffusion(image: Image.Image) -> np.ndarray:
    return np.array(image).astype(np.float32) / 127.5 - 1.0


def recover_with(
    original: Image.Image,
    sampled: Tensor,
    crop: "CropResponse",
    wh_ratio: Tuple[float, float],
    settings: "InpaintingSettings",
) -> Tensor:
    l, t, r, b = crop.lt_rb.tuple
    w_ratio, h_ratio = wh_ratio
    l = round(l * w_ratio)
    t = round(t * h_ratio)
    r = round(r * w_ratio)
    b = round(b * h_ratio)
    c_mask = crop.cropped_mask
    if settings.mask_padding is None:
        c_blurred_mask = c_mask
    else:
        blur = settings.mask_padding
        if isinstance(blur, int):
            blur = blur, blur
        if blur[0] > 0 and blur[1] > 0:
            c_blurred_mask = cv2.blur(c_mask, blur)
        else:
            c_blurred_mask = c_mask
    sampled_array = sampled.numpy().transpose([0, 2, 3, 1])
    o_array = normalize_image_to_diffusion(to_rgb(original))
    c_o_array = ImageBox(l, t, r, b).crop(o_array)
    ch, cw = c_o_array.shape[:2]
    if ch != c_blurred_mask.shape[0] or cw != c_blurred_mask.shape[1]:
        c_blurred_mask = resize(c_blurred_mask, (cw, ch))
    c_blurred_mask = c_blurred_mask[..., None]
    mixed: List[np.ndarray] = []
    for i_sampled in sampled_array:
        i_sampled = resize(i_sampled, (cw, ch))
        i_sampled = i_sampled * c_blurred_mask + c_o_array * (1.0 - c_blurred_mask)
        i_pasted = o_array.copy()
        i_pasted[t:b, l:r] = i_sampled
        mixed.append(i_pasted.transpose([2, 0, 1]))
    return torch.from_numpy(np.stack(mixed, axis=0))


def crop_controlnet(kwargs: Dict[str, Any], crop_res: Optional["CropResponse"]) -> None:
    if crop_res is None:
        return
    hint: Optional[List[Tuple[str, Tensor]]] = kwargs.get(CONTROL_HINT_KEY, None)
    if hint is None:
        return
    for i, h in enumerate(hint):
        hw, hh = crop_res.wh
        h_tensor = crop_res.lt_rb.crop_tensor(h[1])
        h_tensor = F.interpolate(h_tensor, (hh, hw), mode="bilinear")
        hint[i] = h[0], h_tensor


def get_cropped(
    image_res: ReadImageResponse,
    mask_res: ReadImageResponse,
    settings: Optional["InpaintingSettings"],
) -> "CroppedResponse":
    ow, oh = image_res.original_size
    ih, iw = image_res.image.shape[-2:]
    wh_ratio = ow / iw, oh / ih
    if settings is None or settings.mode == InpaintingMode.NORMAL:
        crop_res = None
        cropped_image = image_res.image
        cropped_mask = mask_res.image
    elif settings.mode == InpaintingMode.MASKED:
        crop_res = crop_masked_area(image_res.image, mask_res.image, settings)
        cropped_image = crop_res.resized_image_tensor
        cropped_mask = crop_res.resized_mask_tensor
    else:
        raise ValueError(f"Unknown inpainting mode: {settings.mode}")
    if settings is not None:
        if settings.mask_blur is not None:
            blur = settings.mask_blur
            if isinstance(blur, int):
                blur = blur, blur
            if blur[0] > 0 and blur[1] > 0:
                cropped_mask = cv2.blur(cropped_mask[0][0], blur)
                cropped_mask = cropped_mask[None, None]
    return CroppedResponse(cropped_image, cropped_mask, wh_ratio, crop_res)


def inject_inpainting_sampler_settings(kwargs: Dict[str, Any]) -> None:
    pass


def inject_uncond_indices(hooks: SpatialTransformerHooks, num_samples: int) -> None:
    states = hooks.style_reference_states
    if states is not None:
        states.uncond_indices = list(range(num_samples, num_samples * 2))


class SizeInfo(NamedTuple):
    factor: int
    opt_size: int


class MaskedCond(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    mask_cond: Tensor
    remained_image_cond: Tensor
    remained_image: np.ndarray
    remained_mask: np.ndarray
    image_alpha: Optional[np.ndarray]
    original_size: Tuple[int, int]
    original_image: Image.Image
    original_mask: Image.Image
    wh_ratio: Tuple[float, float]
    crop_res: Optional["CropResponse"]


class CropResponse(NamedTuple):
    lt_rb: ImageBox
    wh: Tuple[int, int]
    cropped_mask: np.ndarray
    resized_image_tensor: np.ndarray
    resized_mask_tensor: np.ndarray


class CroppedResponse(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    wh_ratio: Tuple[float, float]
    crop_res: Optional[CropResponse]


class InpaintingMode(str, Enum):
    NORMAL = "normal"
    MASKED = "masked"


@dataclass
class InpaintingSettings:
    mode: InpaintingMode = InpaintingMode.NORMAL
    mask_blur: TNumberPair = None
    mask_padding: TNumberPair = 32
    mask_binary_threshold: Optional[int] = 32
    target_wh: TNumberPair = None
    padding_mode: Optional[str] = None


class switch_sampler_context:
    def __init__(self, api: "DiffusionAPI", sampler: Optional[str]):
        self.api = api
        self.m_sampler = api.m.sampler.__identifier__
        self.target_sampler = sampler
        self.switched = False

    def __enter__(self) -> None:
        if self.target_sampler is None:
            return
        if self.m_sampler != self.target_sampler:
            self.switched = True
            self.api.switch_sampler(self.target_sampler)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.switched:
            self.api.switch_sampler(self.m_sampler)


class DiffusionAPI(IAPI):
    m: DDPM
    _random_state: Optional[tuple] = None

    def __init__(
        self,
        m: DDPM,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ):
        self.clip_skip = clip_skip
        self.sd_weights = WeightsPool()
        self.current_sd_version: Optional[str] = None
        # extracted the condition model so we can pre-calculate the conditions
        self.cond_model = m.condition_model
        if self.cond_model is not None:
            freeze(self.cond_model)
        m.condition_model = nn.Identity()
        # inference mode flag, should be switched to `False` when `compile`d
        self._use_inference_mode = True
        # inherit
        super().__init__(m, device, use_amp=use_amp, use_half=use_half)

    # core sampling method

    def sample(
        self,
        num_samples: int,
        export_path: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        use_seed_resize: bool = False,
        # each variation contains (seed, weight)
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        z: Optional[Tensor] = None,
        z_ref: Optional[Tensor] = None,
        z_ref_mask: Optional[Tensor] = None,
        z_ref_noise: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        cond_concat: Optional[Tensor] = None,
        unconditional_cond: Optional[Any] = None,
        hint: Optional[Union[Tensor, tensor_dict_type]] = None,
        hint_start: Optional[Union[float, Dict[str, float]]] = None,
        hint_end: Optional[Union[float, Dict[str, float]]] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if self._random_state is None:
            self._random_state = random.getstate()
        o_kw_backup = dict(
            seed=seed,
            variations=variations,
            alpha=alpha,
            cond=cond,
            cond_concat=cond_concat,
            unconditional_cond=unconditional_cond,
            hint=hint,
            hint_start=hint_start,
            hint_end=hint_end,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            batch_size=batch_size,
            verbose=verbose,
        )
        if batch_size is None:
            batch_size = num_samples
        registered_custom = False
        if self.cond_model is not None:
            if isinstance(self.cond_model, CLIPTextConditionModel):
                self.cond_model.clip_skip = kwargs.get(
                    "clip_skip",
                    0 if self.clip_skip is None else self.clip_skip,
                )
            if isinstance(self.cond_model, CLIPTextConditionModel):
                custom_embeddings = kwargs.get("custom_embeddings")
                if custom_embeddings is not None:
                    registered_custom = True
                    self.cond_model.register_custom(custom_embeddings)
        if cond is not None:
            if self.m.condition_type != CONCAT_TYPE and self.cond_model is not None:
                cond = self.get_cond(cond)
        if cond is not None and num_samples != len(cond):
            raise ValueError(
                f"`num_samples` ({num_samples}) should be identical with "
                f"the number of `cond` ({len(cond)})"
            )
        if alpha is not None and original_size is not None:
            alpha_h, alpha_w = alpha.shape[-2:]
            if alpha_w != original_size[0] or alpha_h != original_size[1]:
                raise ValueError(
                    f"shape of the provided `alpha` ({alpha_w}, {alpha_h}) should be "
                    f"identical with the provided `original_size` {original_size}"
                )
        unconditional = cond is None
        if unconditional:
            cond = [0] * num_samples
        cond_data: ArrayData = ArrayData.init(DataConfig.inference_with(batch_size))
        cond_data.fit(cond)
        iterator = TensorBatcher(cond_data.get_loaders()[0], self.device)
        num_iter = len(iterator)
        if verbose and num_iter > 1:
            iterator = tqdm(iterator, desc="iter", total=num_iter)
        sampled = []
        kw = dict(num_steps=num_steps, verbose=verbose)
        kw.update(shallow_copy_dict(kwargs))
        factor, opt_size = self.size_info
        if size is None:
            size = opt_size, opt_size
        else:
            size = tuple(map(lambda n: round(n / factor), size))  # type: ignore
        sampler = self.m.sampler
        uncond_backup = None
        unconditional_cond_backup = None
        if unconditional_cond is None:
            unconditional_cond = getattr(sampler, "unconditional_cond", None)
        if self.cond_model is not None and unconditional_cond is not None:
            uncond_backup = getattr(sampler, "uncond", None)
            unconditional_cond_backup = getattr(
                sampler,
                "unconditional_cond",
                None,
            )
            uncond = self.get_cond(unconditional_cond).to(self.device)
            sampler.uncond = uncond.clone()
            sampler.unconditional_cond = uncond.clone()
        highres_info = kwargs.get("highres_info")
        walk_spatial_transformer_hooks(
            self.m.unet,
            lambda h, _: inject_uncond_indices(h, num_samples),
        )
        with eval_context(self.m, use_inference=self._use_inference_mode):
            with self.amp_context:
                for i, batch in enumerate(iterator):
                    # from the 2nd batch forward, we need to re-generate new seeds
                    if i >= 1:
                        seed = new_seed()
                    i_kw = shallow_copy_dict(kw)
                    i_kw_backup = shallow_copy_dict(i_kw)
                    i_cond = batch[INPUT_KEY]
                    if not unconditional:
                        i_cond = i_cond.to(self.device)
                    i_n = len(i_cond)
                    repeat = (
                        lambda t: t
                        if t.shape[0] == i_n
                        else t.repeat_interleave(i_n, dim=0)
                    )
                    if z is not None:
                        i_z = repeat(z)
                    else:
                        in_channels = self.m.in_channels
                        if self.m.condition_type == CONCAT_TYPE:
                            in_channels -= cond.shape[1]
                        elif cond_concat is not None:
                            in_channels -= cond_concat.shape[1]
                        i_z_shape = i_n, in_channels, *size[::-1]
                        i_z, _ = self._set_seed_and_variations(
                            seed,
                            lambda: torch.randn(i_z_shape, device=self.device),
                            lambda noise: noise,
                            variations,
                            variation_seed,
                            variation_strength,
                        )
                    if use_seed_resize:
                        z_original_shape = list(i_z.shape[-2:])
                        z_opt_shape = list(
                            map(lambda n: round(n / factor), [opt_size, opt_size])
                        )
                        if z_original_shape != z_opt_shape:
                            dx = (z_original_shape[0] - z_opt_shape[0]) // 2
                            dy = (z_original_shape[1] - z_opt_shape[1]) // 2
                            x = z_opt_shape[0] if dx >= 0 else z_opt_shape[0] + 2 * dx
                            y = z_opt_shape[1] if dy >= 0 else z_opt_shape[1] + 2 * dy
                            dx = max(-dx, 0)
                            dy = max(-dy, 0)
                            i_opt_z_shape = (
                                i_n,
                                self.m.in_channels,
                                *z_opt_shape,
                            )
                            i_opt_z, _ = self._set_seed_and_variations(
                                seed,
                                lambda: torch.randn(i_opt_z_shape, device=self.device),
                                lambda noise: noise,
                                variations,
                                variation_seed,
                                variation_strength,
                            )
                            i_z[..., dx : dx + x, dy : dy + y] = i_opt_z[
                                ..., dx : dx + x, dy : dy + y
                            ]
                    if z_ref is not None and z_ref_mask is not None:
                        if z_ref_noise is not None:
                            i_kw["ref"] = repeat(z_ref)
                            i_kw["ref_mask"] = repeat(z_ref_mask)
                            i_kw["ref_noise"] = repeat(z_ref_noise)
                    if unconditional:
                        i_cond = None
                    if self.to_half:
                        i_z = i_z.half()
                        for k, v in i_kw.items():
                            if k.startswith("ref") and not self.use_half:
                                continue
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                i_kw[k] = v.half()
                    if self.use_half:
                        if i_cond is not None:
                            i_cond = i_cond.half()
                    if cond_concat is not None:
                        if self.m.condition_type != HYBRID_TYPE:
                            raise ValueError(
                                f"condition type should be `{HYBRID_TYPE}` when "
                                f"`cond_concat` is provided"
                            )
                        i_cond = {
                            CROSS_ATTN_KEY: i_cond,
                            CONCAT_KEY: cond_concat,
                        }
                    if hint is not None:
                        if isinstance(i_cond, dict):
                            i_cond[CONTROL_HINT_KEY] = hint
                            i_cond[CONTROL_HINT_START_KEY] = hint_start
                            i_cond[CONTROL_HINT_END_KEY] = hint_end
                        else:
                            i_cond = {
                                CROSS_ATTN_KEY: i_cond,
                                CONTROL_HINT_KEY: hint,
                                CONTROL_HINT_START_KEY: hint_start,
                                CONTROL_HINT_END_KEY: hint_end,
                            }
                    with switch_sampler_context(self, i_kw.get("sampler")):
                        if highres_info is not None:
                            # highres workaround
                            i_kw["return_latent"] = True
                        i_inputs = DecoderInputs(z=i_z, cond=i_cond, kwargs=i_kw)
                        i_sampled = self.m.decode(i_inputs)
                        if highres_info is not None:
                            i_z = self._get_highres_latent(i_sampled, highres_info)
                            fidelity = highres_info["fidelity"]
                            if num_steps is None:
                                num_steps = sampler.default_steps
                            i_num_steps = get_highres_steps(num_steps, fidelity)
                            i_kw_backup.pop("highres_info", None)
                            i_kw_backup.update(o_kw_backup)
                            i_kw_backup["fidelity"] = fidelity
                            i_kw_backup["num_steps"] = i_num_steps
                            i_kw_backup["decode_callback"] = self.empty_cuda_cache
                            self.empty_cuda_cache()
                            i_sampled = self._img2img(i_z, export_path, **i_kw_backup)
                            if original_size is not None:
                                upscale_factor = highres_info["upscale_factor"]
                                original_size = (
                                    round(original_size[0] * upscale_factor),
                                    round(original_size[1] * upscale_factor),
                                )
                    sampled.append(i_sampled.cpu().float())
        if uncond_backup is not None:
            sampler.uncond = uncond_backup
        if unconditional_cond_backup is not None:
            sampler.unconditional_cond = unconditional_cond_backup
        concat = torch.cat(sampled, dim=0)
        if clip_output:
            concat = torch.clip(concat, -1.0, 1.0)
        if callback is not None:
            concat = callback(concat)
        if original_size is not None:
            original_size = (max(original_size[0], 1), max(original_size[1], 1))
            with torch.no_grad():
                concat = F.interpolate(
                    concat,
                    original_size[::-1],
                    mode="bilinear",
                )
        if alpha is not None:
            alpha = torch.from_numpy(2.0 * alpha - 1.0)
            if original_size is None:
                with torch.no_grad():
                    alpha = F.interpolate(
                        alpha,
                        concat.shape[-2:],
                        mode="nearest",
                    )
            concat = torch.cat([concat, alpha], dim=1)
        if export_path is not None:
            save_images(concat, export_path)
        self.empty_cuda_cache()
        if registered_custom:
            self.cond_model.clear_custom()
        if self._random_state is not None:
            random.setstate(self._random_state)
            self._random_state = None
        return concat

    # api

    def txt2img(
        self,
        txt: Union[str, List[str]],
        export_path: Optional[str] = None,
        *,
        anchor: int = 64,
        max_wh: int = 512,
        num_samples: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        txt, num_samples = get_txt_cond(txt, num_samples)
        new_size = get_size(size, anchor, max_wh)
        return self.sample(
            num_samples,
            export_path,
            size=new_size,
            original_size=size,
            cond=txt,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    def txt2img_inpainting(
        self,
        txt: Union[str, List[str]],
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        anchor: int = 64,
        max_wh: int = 512,
        num_samples: Optional[int] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        keep_original: bool = False,
        keep_original_num_fade_pixels: Optional[int] = 50,
        use_raw_inpainting: bool = False,
        inpainting_settings: Optional[InpaintingSettings] = None,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        use_background_guidance: bool = False,
        use_reference: bool = False,
        use_background_reference: bool = False,
        reference_fidelity: float = 0.2,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        def get_z_info_from(
            z_ref_pack: Optional[Tuple[Tensor, Tensor, Tensor]], shape: Tuple[int, int]
        ) -> Tuple[Optional[Tensor], Optional[Tuple[int, int]]]:
            if z_ref_pack is None or not use_reference:
                z = None
                size = tuple(map(lambda n: n * self.size_info.factor, shape))
            else:
                size = None
                z_ref, z_ref_mask, z_ref_noise = z_ref_pack
                if use_background_reference:
                    z_ref = z_ref * z_ref_mask + z_ref_noise * (1.0 - z_ref_mask)
                args = z_ref, num_steps, reference_fidelity, seed
                q_sample_kw = shallow_copy_dict(kwargs)
                q_sample_kw["q_sample_noise"] = z_ref_noise
                q_sample_kw["in_inpainting"] = True
                z, _, start_step = self._q_sample(*args, **q_sample_kw)
                kwargs["start_step"] = start_step
            return z, size  # type: ignore

        def paste_original(
            original_: Image.Image,
            mask_: Image.Image,
            sampled_: Tensor,
        ) -> Tensor:
            rgb = to_rgb(original_)
            fade = keep_original_num_fade_pixels
            if not fade:
                rgb_normalized = normalize_image_to_diffusion(rgb)
                rgb_normalized = rgb_normalized.transpose([2, 0, 1])[None]
                mask_res_ = read_image(mask_, None, anchor=None, to_mask=True)
                remained_mask_ = mask_res_.image < 0.5
                pasted = np.where(remained_mask_, rgb_normalized, sampled_.numpy())
                return torch.from_numpy(pasted)
            alpha = to_alpha_channel(mask_)
            fg = Image.merge("RGBA", (*rgb.split(), ImageOps.invert(alpha)))
            sampled_array = sampled_.numpy().transpose([0, 2, 3, 1])
            sampled_uint8_array = ((sampled_array + 1.0) * 127.5).astype(np.uint8)
            merged_arrays = []
            for bg_array in sampled_uint8_array:
                bg = Image.fromarray(bg_array)
                merged = ImageProcessor.paste(fg, bg, num_fade_pixels=fade)
                merged_rgb = merged.convert("RGB")
                merged_arrays.append(normalize_image_to_diffusion(merged_rgb))
            merged_array = np.stack(merged_arrays, axis=0).transpose([0, 3, 1, 2])
            return torch.from_numpy(merged_array).contiguous()

        use_reference = use_reference or use_background_reference
        if inpainting_settings is None:
            inpainting_settings = InpaintingSettings()
        txt_list, num_samples = get_txt_cond(txt, num_samples)

        z_ref_pack: Optional[Tuple[Tensor, Tensor, Tensor]]
        with switch_sampler_context(self, kwargs.get("sampler")):
            # raw inpainting
            if use_raw_inpainting:
                image_res = read_image(
                    image,
                    max_wh,
                    anchor=anchor,
                    padding_mode=inpainting_settings.padding_mode,
                )
                mask_res = read_image(mask, max_wh, anchor=anchor, to_mask=True)
                cropped_res = get_cropped(image_res, mask_res, inpainting_settings)
                z_ref_pack = self._get_z_ref_pack(
                    cropped_res.image, cropped_res.mask, seed
                )
                z_ref, z_ref_mask, z_ref_noise = z_ref_pack
                size: Optional[Tuple[int, ...]]
                if use_reference:
                    z, size = get_z_info_from(z_ref_pack, z_ref.shape[-2:][::-1])
                else:
                    z = None
                    size = cropped_res.image.shape[-2:][::-1]
                kw = shallow_copy_dict(kwargs)
                kw.update(
                    dict(
                        z=z,
                        size=size,
                        z_ref=z_ref,
                        z_ref_mask=z_ref_mask,
                        z_ref_noise=z_ref_noise,
                        original_size=image_res.original_size,
                        alpha=None,
                        cond=txt_list,
                        num_steps=num_steps,
                        clip_output=clip_output,
                        verbose=verbose,
                    )
                )
                inject_inpainting_sampler_settings(kw)
                crop_controlnet(kw, cropped_res.crop_res)
                sampled = self.sample(num_samples, **kw)
                crop_res = cropped_res.crop_res
                if crop_res is not None:
                    sampled = recover_with(
                        image_res.original,
                        sampled,
                        crop_res,
                        cropped_res.wh_ratio,
                        inpainting_settings,
                    )
                if keep_original:
                    original = image_res.original
                    sampled = paste_original(original, mask_res.original, sampled)
                if export_path is not None:
                    save_images(sampled, export_path)
                return sampled

            # 'real' inpainting
            res = self._get_masked_cond(
                image,
                mask,
                max_wh,
                anchor,
                lambda remained_mask, img: np.where(remained_mask, img, 0.5),
                lambda bool_mask: torch.from_numpy(bool_mask),
                inpainting_settings,
            )
            # sampling
            ## calculate `z_ref` stuffs, if needed
            if not use_reference and not use_background_guidance:
                z_ref_pack = z_ref = z_ref_mask = z_ref_noise = None
            else:
                z_ref_pack = self._get_z_ref_pack(res.image, res.mask, seed)
                z_ref, z_ref_mask, z_ref_noise = z_ref_pack
            ## calculate `z` based on `z_ref`, if needed
            z_shape = res.remained_image_cond.shape[-2:][::-1]
            z, size = get_z_info_from(z_ref_pack, z_shape)
            ## adjust ControlNet parameters
            crop_controlnet(kwargs, res.crop_res)
            ## core
            sampled = self.sample(
                num_samples,
                seed=seed,
                z=z,
                z_ref=z_ref,
                z_ref_mask=z_ref_mask,
                z_ref_noise=z_ref_noise,
                size=size,  # type: ignore
                original_size=res.original_size,
                alpha=None,
                cond=txt_list,
                cond_concat=torch.cat([res.mask_cond, res.remained_image_cond], dim=1),
                num_steps=num_steps,
                clip_output=clip_output,
                callback=callback,
                verbose=verbose,
                **kwargs,
            )
            if res.crop_res is not None:
                sampled = recover_with(
                    res.original_image,
                    sampled,
                    res.crop_res,
                    res.wh_ratio,
                    inpainting_settings,
                )
        if keep_original:
            sampled = paste_original(res.original_image, res.original_mask, sampled)
        if export_path is not None:
            save_images(sampled, export_path)
        return sampled

    def outpainting(
        self,
        txt: str,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 64,
        max_wh: int = 512,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        keep_original: bool = False,
        keep_original_num_fade_pixels: Optional[int] = 50,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGBA":
            raise ValueError("`image` should be `RGBA` in outpainting")
        *rgb, alpha = image.split()
        mask = Image.fromarray(255 - np.array(alpha))
        image = Image.merge("RGB", rgb)
        return self.txt2img_inpainting(
            txt,
            image,
            mask,
            export_path,
            anchor=anchor,
            max_wh=max_wh,
            num_steps=num_steps,
            clip_output=clip_output,
            keep_original=keep_original,
            keep_original_num_fade_pixels=keep_original_num_fade_pixels,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    def img2img(
        self,
        image: Union[str, Image.Image, Tensor],
        export_path: Optional[str] = None,
        *,
        anchor: int = 32,
        max_wh: int = 512,
        fidelity: float = 0.2,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if isinstance(image, Tensor):
            original_size = tuple(image.shape[-2:][::-1])
        else:
            res = read_image(image, max_wh, anchor=anchor)
            image = res.image
            original_size = res.original_size
            if alpha is None:
                alpha = res.alpha
        z = self._get_z(image)
        highres_info = kwargs.pop("highres_info", None)
        if highres_info is not None:
            z = self._get_highres_latent(z, highres_info)
            if num_steps is None:
                num_steps = self.m.sampler.default_steps
            num_steps = get_highres_steps(num_steps, fidelity)
            upscale_factor = highres_info["upscale_factor"]
            original_size = (
                round(original_size[0] * upscale_factor),
                round(original_size[1] * upscale_factor),
            )
            if alpha is not None:
                with torch.no_grad():
                    alpha = F.interpolate(
                        torch.from_numpy(alpha),
                        original_size[::-1],
                        mode="nearest",
                    ).numpy()
        return self._img2img(
            z,
            export_path,
            fidelity=fidelity,
            original_size=original_size,  # type: ignore
            alpha=alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def inpainting(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 32,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        refine_fidelity: Optional[float] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # inpainting callback, will not trigger in refine stage
        def callback(out: Tensor) -> Tensor:
            final = torch.from_numpy(res.remained_image.copy())
            final += 0.5 * (1.0 + out) * (1.0 - res.remained_mask)
            return 2.0 * final - 1.0

        res = self._get_masked_cond(
            image,
            mask,
            max_wh,
            anchor,
            lambda remained_mask, img: np.where(remained_mask, img, 0.0),
            lambda bool_mask: torch.where(torch.from_numpy(bool_mask), 1.0, -1.0),
        )
        cond = torch.cat([res.remained_image_cond, res.mask_cond], dim=1)
        size = self._get_identical_size_with(res.remained_image_cond)
        # refine with img2img
        if refine_fidelity is not None:
            z = self._get_z(res.image)
            return self._img2img(
                z,
                export_path,
                fidelity=refine_fidelity,
                original_size=res.original_size,
                alpha=res.image_alpha if alpha is None else alpha,
                cond=cond,
                num_steps=num_steps,
                clip_output=clip_output,
                verbose=verbose,
                **kwargs,
            )
        # sampling
        return self.sample(
            1,
            export_path,
            size=size,  # type: ignore
            original_size=res.original_size,
            alpha=res.image_alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    def semantic2img(
        self,
        semantic: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 16,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        err_fmt = "`{}` is needed for `semantic2img`"
        if self.cond_model is None:
            raise ValueError(err_fmt.format("cond_model"))
        in_channels = getattr(self.cond_model, "in_channels", None)
        if in_channels is None:
            raise ValueError(err_fmt.format("cond_model.in_channels"))
        factor = getattr(self.cond_model, "factor", None)
        if factor is None:
            raise ValueError(err_fmt.format("cond_model.factor"))
        res = read_image(
            semantic,
            max_wh,
            anchor=anchor,
            to_gray=True,
            resample=Image.NEAREST,
            normalize=False,
        )
        cond = torch.from_numpy(res.image).to(torch.long).to(self.device)
        cond = F.one_hot(cond, num_classes=in_channels)[0]
        cond = cond.half() if self.to_half else cond.float()
        cond = cond.permute(0, 3, 1, 2).contiguous()
        cond = self.get_cond(cond)
        size = self._get_identical_size_with(cond)
        return self.sample(
            1,
            export_path,
            size=size,
            original_size=res.original_size,
            alpha=res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    # utils

    @property
    def first_stage(self) -> Optional[nn.Module]:
        if isinstance(self.m, LDM):
            return self.m.first_stage
        return None

    @property
    def size_info(self) -> SizeInfo:
        opt_size = self.m.img_size
        first_stage = self.first_stage
        if first_stage is None:
            factor = 1
        else:
            factor = first_stage.img_size // opt_size
        return SizeInfo(factor, opt_size)

    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        super().to(device, use_amp=use_amp, use_half=use_half)
        unconditional_cond = getattr(self.m.sampler, "unconditional_cond", None)
        if not isinstance(unconditional_cond, Tensor):
            unconditional_cond = None
        if self.to_half:
            if self.cond_model is not None:
                self.cond_model.half()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.half()
        else:
            if self.cond_model is not None:
                self.cond_model.float()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.float()
        if self.cond_model is not None:
            self.cond_model.to(device)
        if unconditional_cond is not None:
            self.m.sampler.unconditional_cond = unconditional_cond.to(device)

    def compile(self, *, compile_cond: bool = False, **kwargs: Any) -> "DiffusionAPI":
        self.m = torch.compile(self.m, **kwargs)
        if compile_cond and self.cond_model is not None:
            self.cond_model = torch.compile(self.cond_model, **kwargs)
        self._use_inference_mode = False
        return self

    def prepare_sd(
        self,
        versions: List[str],
        *,
        # inpainting workarounds
        # should set `force_external` to `True` to prepare inpainting with this method
        sub_folder: Optional[str] = None,
        force_external: bool = False,
    ) -> None:
        for tag in map(get_sd_tag, versions):
            if tag not in self.sd_weights:
                _load_external = lambda: convert_external(self, tag, sub_folder)
                if force_external:
                    model_path = _load_external()
                else:
                    try:
                        model_path = download_checkpoint(f"ldm_sd_{tag}")
                    except:
                        model_path = _load_external()
                self.sd_weights.register(tag, model_path)

    def switch_sd(self, version: str) -> None:
        tag = get_sd_tag(version)
        if self.current_sd_version is not None:
            if tag == get_sd_tag(self.current_sd_version):
                return
        d = self.sd_weights.get(tag)
        with self.load_context() as m:
            m.load_state_dict(d)
        self.current_sd_version = version

    def switch_sampler(
        self,
        sampler: str,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        sampler_ins = self.m.make_sampler(sampler, sampler_config)
        current_unconditional_cond = getattr(self.m.sampler, "unconditional_cond", None)
        if current_unconditional_cond is not None:
            if hasattr(sampler_ins, "unconditional_cond"):
                sampler_ins.unconditional_cond = current_unconditional_cond
        current_guidance = getattr(self.m.sampler, "unconditional_guidance_scale", None)
        if current_guidance is not None:
            if hasattr(sampler_ins, "unconditional_guidance_scale"):
                sampler_ins.unconditional_guidance_scale = current_guidance
        self.m.sampler = sampler_ins

    def switch_circular(self, enable: bool) -> None:
        def _inject(m: nn.Module) -> None:
            for child in m.children():
                _inject(child)
            modules.append(m)

        padding_mode = "circular" if enable else "zeros"
        modules: List[nn.Module] = []
        _inject(self.m)
        for module in modules:
            if isinstance(module, nn.Conv2d):
                module.padding_mode = padding_mode
            elif isinstance(module, Conv2d):
                module.padding = padding_mode

    def get_cond(self, cond: Any) -> Tensor:
        if self.cond_model is None:
            msg = "should not call `get_cond` when `cond_model` is not available"
            raise ValueError(msg)
        with torch.no_grad():
            with self.amp_context:
                return self.cond_model(cond)

    def load_context(self, *, ignore_lora: bool = True) -> ContextManager:
        class _:
            def __init__(self, api: DiffusionAPI):
                self.api = api
                self.m_ctrl = api.m.control_model
                self.m_cond = api.m.condition_model
                api.m.control_model = None
                api.m.condition_model = api.cond_model
                if not ignore_lora:
                    self.lora_checkpoints = None
                else:
                    if not isinstance(api.m, StableDiffusion):
                        msg = "currently only `StableDiffusion` supports `ignore_lora`"
                        raise ValueError(msg)
                    if not api.m.has_lora:
                        self.lora_checkpoints = None
                    else:
                        self.lora_checkpoints = api.m.get_lora_checkpoints()
                        api.m.cleanup_lora()

            def __enter__(self) -> DDPM:
                return self.api.m

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self.lora_checkpoints is not None:
                    assert isinstance(self.api.m, StableDiffusion)
                    self.api.m.restore_lora_from(self.lora_checkpoints)
                self.api.m.control_model = self.m_ctrl
                self.api.m.condition_model = self.m_cond

        return _(self)

    # lora

    def load_sd_lora(self, key: str, *, path: str) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `load_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.load_lora(key, path=path)

    def inject_sd_lora(self, *keys: str) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `inject_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.inject_lora(*keys)

    def cleanup_sd_lora(self) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `cleanup_sd_lora`")
        with self.load_context(ignore_lora=False):
            self.m.cleanup_lora()

    def set_sd_lora_scales(self, scales: Dict[str, float]) -> None:
        if not isinstance(self.m, StableDiffusion):
            raise ValueError("only `StableDiffusion` can use `set_sd_lora_scales`")
        with self.load_context(ignore_lora=False):
            self.m.set_lora_scales(scales)

    # hooks

    def setup_hooks(
        self,
        *,
        tome_info: Optional[Dict[str, Any]] = None,
        style_reference_image: Optional[Union[str, Image.Image]] = None,
        style_reference_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        def mutate_states(hooks: SpatialTransformerHooks) -> None:
            if style_reference_image is None:
                return
            if hooks.style_reference_states is None:
                return

            def get_ref_net(timesteps: Tensor) -> Tensor:
                timesteps = torch.round(timesteps.float()).long()
                return self.m.q_sampler.q_sample(ref_z, timesteps)

            ref_image = read_image(style_reference_image, None, anchor=64).image
            ref_z = self._get_z(ref_image)
            hooks.style_reference_states.get_ref_net = get_ref_net

        unet = self.m.unet
        unet.setup_hooks(
            tome_info=tome_info,
            style_reference_states=style_reference_states,
        )
        walk_spatial_transformer_hooks(unet, lambda hooks, _: mutate_states(hooks))

    # constructors

    @classmethod
    def from_sd(
        cls: Type[T],
        version: Optional[str] = None,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ) -> T:
        m = cls(
            ldm_sd(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=clip_skip,
        )
        if version is not None:
            m.prepare_sd([version])
            m.switch_sd(version)
        return m

    @classmethod
    def from_sd_inpainting(
        cls: Type[T],
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ) -> T:
        return cls(
            ldm_sd_inpainting(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=clip_skip,
        )

    # internal

    def _get_z(self, img: arr_type) -> Tensor:
        img = 2.0 * img - 1.0
        z = img if isinstance(img, Tensor) else torch.from_numpy(img)
        if self.to_half:
            z = z.half()
        z = z.to(self.device)
        z = self.m.preprocess(z, deterministic=True)
        return z

    def _get_z_ref_pack(
        self,
        image_tensor: np.ndarray,
        mask_tensor: np.ndarray,
        seed: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        z_ref = self._get_z(image_tensor)
        z_ref_mask = 1.0 - F.interpolate(
            torch.from_numpy(mask_tensor).to(z_ref),
            z_ref.shape[-2:],
            mode="bilinear",
        )
        if seed is not None:
            seed_everything(seed)
        z_ref_noise = torch.randn_like(z_ref)
        return z_ref, z_ref_mask, z_ref_noise

    def _get_identical_size_with(self, pivot: Tensor) -> Tuple[int, int]:
        return tuple(  # type: ignore
            map(
                lambda n: n * self.size_info.factor,
                pivot.shape[-2:][::-1],
            )
        )

    def _set_seed_and_variations(
        self,
        seed: Optional[int],
        get_noise: Callable[[], Tensor],
        get_new_z: Callable[[Tensor], Tensor],
        variations: Optional[List[Tuple[int, float]]],
        variation_seed: Optional[int],
        variation_strength: Optional[float],
    ) -> Tuple[Tensor, Tensor]:
        if seed is None:
            seed = new_seed()
        seed = seed_everything(seed)
        self.latest_seed = seed
        z_noise = get_noise()
        self.latest_variation_seed = None
        if variations is not None:
            for v_seed, v_weight in variations:
                seed_everything(v_seed)
                v_noise = get_noise()
                z_noise = slerp(v_noise, z_noise, v_weight)
        if variation_strength is not None:
            random.seed()
            if variation_seed is None:
                variation_seed = new_seed()
            variation_seed = seed_everything(variation_seed)
            self.latest_variation_seed = variation_seed
            variation_noise = get_noise()
            z_noise = slerp(variation_noise, z_noise, variation_strength)
        z = get_new_z(z_noise)
        return z, z_noise

    def _get_masked_cond(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        max_wh: int,
        anchor: int,
        mask_image_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mask_cond_fn: Callable[[np.ndarray], Tensor],
        inpainting_settings: Optional[InpaintingSettings] = None,
    ) -> MaskedCond:
        # handle mask stuffs
        mask_res = read_image(mask, max_wh, anchor=anchor, to_mask=True)
        read_image_kw = {}
        if inpainting_settings is not None:
            if inpainting_settings.padding_mode is not None:
                o_mask = mask_res.original
                padding_mask = to_alpha_channel(o_mask)
                read_image_kw["padding_mask"] = padding_mask
                read_image_kw["padding_mode"] = inpainting_settings.padding_mode
        image_res = read_image(image, max_wh, anchor=anchor, **read_image_kw)
        cropped_res = get_cropped(image_res, mask_res, inpainting_settings)
        c_image = cropped_res.image
        c_mask = cropped_res.mask
        bool_mask = np.round(c_mask) >= 0.5
        remained_mask = (~bool_mask).astype(np.float16 if self.to_half else np.float32)
        remained_image = mask_image_fn(remained_mask, c_image)
        # construct condition tensor
        remained_cond = self._get_z(remained_image)
        latent_shape = remained_cond.shape[-2:]
        mask_cond = mask_cond_fn(bool_mask).to(torch.float32)
        mask_cond = F.interpolate(mask_cond, size=latent_shape)
        if self.to_half:
            mask_cond = mask_cond.half()
        mask_cond = mask_cond.to(self.device)
        return MaskedCond(
            c_image,
            c_mask,
            mask_cond,
            remained_cond,
            remained_image,
            remained_mask,
            image_res.alpha,
            image_res.original_size,
            image_res.original,
            mask_res.original,
            cropped_res.wh_ratio,
            cropped_res.crop_res,
        )

    def _q_sample(
        self,
        z: Tensor,
        num_steps: Optional[int],
        fidelity: float,
        seed: Optional[int],
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        q_sample_noise: Optional[Tensor] = None,
        in_inpainting: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, int]:
        sampler = self.m.sampler
        if self._random_state is None:
            self._random_state = random.getstate()
        if num_steps is None:
            num_steps = sampler.default_steps
        if isinstance(sampler, IKSampler):
            t = min(num_steps, round((1.0 - fidelity) * (num_steps + 1))) - 1
            ts = get_timesteps(t + 1, 1, z.device)
            start_step = num_steps - t - 1
        else:
            t = int(min(1.0 - fidelity, 0.999) * num_steps)
            ts = get_timesteps(t, 1, z.device)
            start_step = num_steps - t
        if isinstance(sampler, (DDIMMixin, IKSampler, DPMSolver)):
            kw = shallow_copy_dict(sampler.sample_kwargs)
            kw.update(shallow_copy_dict(kwargs))
            kw["total_step"] = num_steps
            if in_inpainting:
                inject_inpainting_sampler_settings(kw)
            safe_execute(sampler._reset_buffers, kw)
        if q_sample_noise is not None:
            z = sampler.q_sample(z, ts, q_sample_noise)
            noise = q_sample_noise
        else:
            z, noise = self._set_seed_and_variations(
                seed,
                lambda: torch.randn_like(z),
                lambda noise: sampler.q_sample(z, ts, noise),
                variations,
                variation_seed,
                variation_strength,
            )
        return z, noise, start_step

    def _img2img(
        self,
        z: Tensor,
        export_path: Optional[str] = None,
        *,
        z_ref: Optional[Tensor] = None,
        z_ref_mask: Optional[Tensor] = None,
        original_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        with switch_sampler_context(self, kwargs.get("sampler")):
            z, noise, start_step = self._q_sample(z, num_steps, **kwargs)
            kwargs["start_step"] = start_step
            return self.sample(
                z.shape[0],
                export_path,
                z=z,
                z_ref=z_ref,
                z_ref_mask=z_ref_mask,
                z_ref_noise=None if z_ref is None else noise,
                original_size=original_size,
                alpha=alpha,
                cond=cond,
                num_steps=num_steps,
                clip_output=clip_output,
                verbose=verbose,
                **kwargs,
            )

    def _get_highres_latent(self, z: Tensor, highres_info: Dict[str, Any]) -> Tensor:
        upscale_factor = highres_info["upscale_factor"]
        shrink_factor = self.size_info.factor
        max_wh = round(highres_info["max_wh"] / shrink_factor)
        h, w = z.shape[-2:]
        upscaled = round(w * upscale_factor), round(h * upscale_factor)
        w, h = get_size(upscaled, 64 // shrink_factor, max_wh)  # type: ignore
        return F.interpolate(z, size=(h, w), mode="bilinear", antialias=False)


# controlnet


class ControlNetHints(str, Enum):
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    MLSD = "mlsd"
    SOFTEDGE = "softedge"


class ControlledDiffusionAPI(DiffusionAPI):
    loaded: Dict[ControlNetHints, bool]
    annotators: Dict[Union[str, ControlNetHints], Annotator]
    controlnet_weights: Dict[ControlNetHints, tensor_dict_type]
    controlnet_latest_usage: Dict[ControlNetHints, float]

    control_mappings = {
        ControlNetHints.DEPTH: "ldm.sd_v1.5.control.diff.depth",
        ControlNetHints.CANNY: "ldm.sd_v1.5.control.diff.canny",
        ControlNetHints.POSE: "ldm.sd_v1.5.control.diff.pose",
        ControlNetHints.MLSD: "ldm.sd_v1.5.control.diff.mlsd",
    }

    def __init__(
        self,
        m: DDPM,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
        hint_channels: int = 3,
        num_pool: Optional[Union[str, int]] = "all",
        lazy: bool = False,
    ):
        default_cnet = sorted(self.control_mappings)[0]
        self.lazy = lazy
        if num_pool is None:
            self.num_pool = None
        else:
            if num_pool == "all":
                num_pool = len(ControlNetHints)
            self.num_pool = num_pool if isinstance(num_pool, int) else None
        self.hint_channels = hint_channels
        m.make_control_net({default_cnet: hint_channels}, lazy)
        assert isinstance(m.control_model, nn.ModuleDict)
        self.control_model = m.control_model
        freeze(m.control_model)
        self.loaded = {}
        self.annotators = {}
        self.control_model = m.control_model
        self.controlnet_weights = {}
        self.controlnet_latest_usage = {}
        # inherit
        super().__init__(
            m,
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=clip_skip,
        )

    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        no_annotator: bool = False,
    ) -> None:
        super().to(device, use_amp=use_amp, use_half=use_half)
        if not no_annotator and not self.lazy:
            for annotator in self.annotators.values():
                self._annotator_to(annotator)

    @property
    def preset_control_hints(self) -> List[ControlNetHints]:
        return list(self.control_mappings)

    @property
    def available_control_hints(self) -> List[ControlNetHints]:
        return list(self.controlnet_weights)

    def setup_hooks(
        self,
        style_reference_image: Optional[Union[str, Image.Image]] = None,
        **hooks_kwargs: Any,
    ) -> None:
        super().setup_hooks(style_reference_image=style_reference_image, **hooks_kwargs)
        if self.control_model is not None:
            if isinstance(self.control_model, ControlNet):
                self.control_model.setup_hooks(**hooks_kwargs)
            else:
                for m in self.control_model.values():
                    m.setup_hooks(**hooks_kwargs)

    def prepare_control(self, hints2tags: Dict[ControlNetHints, str]) -> None:
        any_new = False
        for hint, tag in hints2tags.items():
            if hint not in self.control_model:
                any_new = True
                self.m.make_control_net(self.hint_channels, self.lazy, target_key=hint)
            if hint not in self.controlnet_weights:
                try:
                    d = torch.load(download_checkpoint(tag))
                except:

                    def fn(p: str, m: ControlledDiffusionAPI) -> tensor_dict_type:
                        import cflearn

                        return cflearn.scripts.sd.convert_controlnet(p)

                    p = convert_external(self, tag, "controlnet", convert_fn=fn)
                    d = torch.load(p)
                self.loaded[hint] = False
                self.controlnet_weights[hint] = d
            elif hint not in self.loaded:
                self.loaded[hint] = False
        if any_new:
            freeze(self.m.control_model)

    def remove_control(self, hints: List[ControlNetHints]) -> None:
        for hint in hints:
            if hint in self.loaded:
                del self.loaded[hint]
            if hint in self.controlnet_weights:
                del self.controlnet_weights[hint]
            if hint in self.controlnet_latest_usage:
                del self.controlnet_latest_usage[hint]
            if hint in self.control_model:
                m = self.control_model.pop(hint)
                m.to("cpu")
                del m

    def switch_control(self, *hints: ControlNetHints) -> None:
        if self.m.control_model is None:
            raise ValueError("`control_model` is not built yet")

        for hint in hints:
            self.controlnet_latest_usage[hint] = time.time()

        target = set(hints)
        # if `hint` does not exist in `control_mappings`, it means it is an
        # external controlnet, so it should be left as-is
        target_mapping = {h: self.control_mappings.get(h, h) for h in target}
        self.prepare_control(target_mapping)

        current = set(self.control_model.keys())
        if self.num_pool is None or len(current) > self.num_pool:
            to_remove = list(current - target)
            if to_remove:
                if self.num_pool is None:
                    print_warning(
                        "`num_pool` is set to `None`, redundant controlnets "
                        f"({to_remove}) will be removed"
                    )
                    self.remove_control(to_remove)
                else:
                    usages = [self.controlnet_latest_usage.get(h, 0) for h in to_remove]
                    sorted_indices = np.argsort(usages)
                    diff = len(current) - self.num_pool
                    remove_indices = sorted_indices[:diff]
                    to_remove = [to_remove[i] for i in remove_indices]
                    print_warning(
                        "current number of controlnets exceeds `num_pool` "
                        f"({self.num_pool}), {to_remove} will be removed"
                    )
                    self.remove_control(to_remove)

        sorted_target = sorted(target)
        loaded_list = [self.loaded[hint] for hint in sorted_target]
        if all(loaded_list):
            return
        iterator = zip(sorted_target, loaded_list)
        for hint, loaded in iterator:
            if loaded:
                continue
            d = self.controlnet_weights.get(hint)
            if d is None:
                raise ValueError(
                    f"cannot find ControlNet weights called '{hint}', "
                    f"available weights are: {', '.join(self.available_control_hints)}"
                )
            self.m.load_control_net_with(hint, d)
            self.loaded[hint] = True

    def prepare_annotator(self, hint: Union[str, ControlNetHints]) -> None:
        if hint not in self.annotators:
            annotator_class = annotators.get(hint)
            if annotator_class is None:
                print_warning(f"annotator '{hint}' is not implemented")
                return
            if self.lazy:
                annotator = annotator_class("cpu")
            else:
                annotator = annotator_class(self.device)
                self._annotator_to(annotator)
            self.annotators[hint] = annotator

    def prepare_annotators(self) -> None:
        for hint in ControlNetHints:
            self.prepare_annotator(hint)

    def get_hint_of(
        self,
        hint: Union[str, ControlNetHints],
        uint8_rgb: np.ndarray,
        *,
        binarize_threshold: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        self.prepare_annotator(hint)
        annotator = self.annotators.get(hint)
        if annotator is None:
            return uint8_rgb
        if self.lazy:
            self._annotator_to(annotator)
        kwargs["uint8_rgb"] = uint8_rgb
        out = safe_execute(annotator.annotate, kwargs)
        if len(out.shape) == 2:
            out = out[..., None]
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=2)
        if binarize_threshold is not None:
            out = np.where(out > binarize_threshold, 255, 0).astype(np.uint8)
        if self.lazy:
            self._annotator_to(annotator, "cpu")
        return out

    def enable_control(self) -> None:
        self.m.control_model = self.control_model

    def disable_control(self) -> None:
        self.m.control_model = None

    # internal

    def _annotator_to(self, annotator: Annotator, device: Optional[str] = None) -> None:
        if device is None:
            device = self.device
        annotator.to(device, use_half=self.use_half)


__all__ = [
    "InpaintingMode",
    "InpaintingSettings",
    "DiffusionAPI",
    "ControlNetHints",
    "ControlledDiffusionAPI",
]
