import json
import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from typing import ContextManager
from pathlib import Path
from filelock import FileLock
from cftool.cv import save_images
from cftool.cv import restrict_wh
from cftool.cv import get_suitable_size
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.types import tensor_dict_type

from ..common import IAPI
from ..common import WeightsPool
from ...zoo import ldm_sd
from ...zoo import get_sd_tag
from ...data import predict_array_data
from ...data import ArrayData
from ...data import TensorBatcher
from ...schema import device_type
from ...schema import DataConfig
from ...modules import LDM
from ...modules import DDPM
from ...modules import Conv2d
from ...modules import StableDiffusion
from ...modules import DDIMMixin
from ...modules import DPMSolver
from ...modules import KSamplerMixin
from ...toolkit import _get_file_size
from ...toolkit import slerp
from ...toolkit import freeze
from ...toolkit import new_seed
from ...toolkit import load_file
from ...toolkit import seed_everything
from ...toolkit import download_checkpoint
from ...toolkit import eval_context
from ...constants import INPUT_KEY
from ...constants import PREDICTIONS_KEY
from ...parameters import OPT
from ...modules.multimodal.diffusion.utils import CONCAT_KEY
from ...modules.multimodal.diffusion.utils import CONCAT_TYPE
from ...modules.multimodal.diffusion.utils import HYBRID_TYPE
from ...modules.multimodal.diffusion.utils import CROSS_ATTN_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_END_KEY
from ...modules.multimodal.diffusion.utils import CONTROL_HINT_START_KEY
from ...modules.multimodal.diffusion.utils import get_timesteps
from ...modules.multimodal.diffusion.cond_models import CLIPTextConditionModel


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


class SizeInfo(NamedTuple):
    factor: int
    opt_size: int


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
        self.m = freeze(m)
        self.setup(device, use_amp, use_half)
        self.clip_skip = clip_skip
        self.sd_weights = WeightsPool()
        self.current_sd_version: Optional[str] = None
        # extracted the condition model so we can pre-calculate the conditions
        self.cond_model = m.condition_model
        if self.cond_model is not None:
            freeze(self.cond_model)
        m.condition_model = nn.Identity()
        # extract first stage
        if not isinstance(m, LDM):
            self.first_stage = None
        else:
            self.first_stage = m.first_stage
        # pre-calculate unconditional_cond if needed
        self._original_raw_uncond = getattr(m.sampler, "unconditional_cond", None)
        self._uncond_cache: tensor_dict_type = {}
        self._update_sampler_uncond(clip_skip)
        # inference mode flag, should be switched to `False` when `compile`d
        self._use_inference_mode = True
        # finalize
        self.to(device, use_amp=use_amp, use_half=use_half)

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
            clip_skip = kwargs.get(
                "clip_skip",
                0 if self.clip_skip is None else self.clip_skip,
            )
            self._update_sampler_uncond(clip_skip)
            if isinstance(self.cond_model, CLIPTextConditionModel):
                custom_embeddings = kwargs.get("custom_embeddings")
                if custom_embeddings is not None:
                    registered_custom = True
                    self.cond_model.register_custom(custom_embeddings)
        if cond is not None:
            if self.m.condition_type != CONCAT_TYPE and self.cond_model is not None:
                cond = predict_array_data(
                    self.cond_model,
                    ArrayData.init().fit(np.array(cond)),
                    batch_size=batch_size,
                )[PREDICTIONS_KEY]
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
        cond_data: ArrayData = ArrayData.init(DataConfig(batch_size=batch_size))
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
        uncond_backup = None
        unconditional_cond_backup = None
        if self.cond_model is not None and unconditional_cond is not None:
            uncond_backup = getattr(self.sampler, "uncond", None)
            unconditional_cond_backup = getattr(
                self.sampler,
                "unconditional_cond",
                None,
            )
            uncond = self.get_cond(unconditional_cond).to(self.device)
            self.sampler.uncond = uncond.clone()
            self.sampler.unconditional_cond = uncond.clone()
        highres_info = kwargs.get("highres_info")
        with eval_context(self.m, use_inference=self._use_inference_mode):
            with self.amp_context:
                for i, batch in enumerate(iterator):
                    # from the 2nd batch forward, we need to re-generate new seeds
                    if i >= 1:
                        seed = new_seed()
                    i_kw = shallow_copy_dict(kw)
                    i_kw_backup = shallow_copy_dict(i_kw)
                    i_cond = batch[INPUT_KEY].to(self.device)
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
                    if self.use_half:
                        i_z = i_z.half()
                        if i_cond is not None:
                            i_cond = i_cond.half()
                        for k, v in i_kw.items():
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                i_kw[k] = v.half()
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
                        i_sampled = self.m.decode(i_z, cond=i_cond, **i_kw)
                        if highres_info is not None:
                            i_z = self._get_highres_latent(i_sampled, highres_info)
                            fidelity = highres_info["fidelity"]
                            if num_steps is None:
                                num_steps = self.sampler.default_steps
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
            self.sampler.uncond = uncond_backup
        if unconditional_cond_backup is not None:
            self.sampler.unconditional_cond = unconditional_cond_backup
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
                    mode="bicubic",
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

    # utils

    @property
    def size_info(self) -> SizeInfo:
        opt_size = self.m.img_size
        if self.first_stage is None:
            factor = 1
        else:
            factor = self.first_stage.img_size // opt_size
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
        if use_half:
            if self.cond_model is not None:
                self.cond_model.half()
            if self.first_stage is not None:
                self.first_stage.half()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.half()
            for k, v in self._uncond_cache.items():
                self._uncond_cache[k] = v.half()
        else:
            if self.cond_model is not None:
                self.cond_model.float()
            if self.first_stage is not None:
                self.first_stage.float()
            if unconditional_cond is not None:
                unconditional_cond = unconditional_cond.float()
            for k, v in self._uncond_cache.items():
                self._uncond_cache[k] = v.float()
        if self.cond_model is not None:
            self.cond_model.to(device)
        if self.first_stage is not None:
            self.first_stage.to(device)
        if unconditional_cond is not None:
            self.m.sampler.unconditional_cond = unconditional_cond.to(device)
        for k, v in self._uncond_cache.items():
            self._uncond_cache[k] = v.to(device)

    def compile(self, **kwargs: Any) -> "DiffusionAPI":
        self.first_stage = torch.compile(self.first_stage, **kwargs)
        self.m = torch.compile(self.m, **kwargs)
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
        self.sampler = self.m.sampler = sampler_ins

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

    # tomesd

    def set_tome_info(self, tome_info: Optional[Dict[str, Any]]) -> None:
        self.m.unet.set_tome_info(tome_info)

    # constructors

    @classmethod
    def from_sd(
        cls,
        version: Optional[str] = None,
        device: device_type = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ) -> "DiffusionAPI":
        return cls(
            ldm_sd(version),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=clip_skip,
        )

    # internal

    def _update_sampler_uncond(self, clip_skip: int) -> None:
        if isinstance(self.cond_model, CLIPTextConditionModel):
            self.cond_model.clip_skip = clip_skip
        if self.cond_model is not None and self._original_raw_uncond is not None:
            cache = self._uncond_cache.get(clip_skip)
            if cache is not None:
                uncond = cache
            else:
                uncond = self.get_cond(self._original_raw_uncond)
                self._uncond_cache[clip_skip] = uncond
            self.m.sampler.unconditional_cond = uncond.to(self.device)

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

    def _q_sample(
        self,
        z: Tensor,
        num_steps: Optional[int],
        fidelity: float,
        seed: Optional[int],
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, int]:
        if self._random_state is None:
            self._random_state = random.getstate()
        if num_steps is None:
            num_steps = self.sampler.default_steps
        t = min(num_steps, round((1.0 - fidelity) * (num_steps + 1)))
        ts = get_timesteps(t, 1, z.device)
        if isinstance(self.sampler, (DDIMMixin, KSamplerMixin, DPMSolver)):
            kw = shallow_copy_dict(self.sampler.sample_kwargs)
            kw["total_step"] = num_steps
            safe_execute(self.sampler._reset_buffers, kw)
        z, noise = self._set_seed_and_variations(
            seed,
            lambda: torch.randn_like(z),
            lambda noise_: self.sampler.q_sample(z, ts, noise_),
            variations,
            variation_seed,
            variation_strength,
        )
        start_step = num_steps - t
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


__all__ = [
    "DiffusionAPI",
]
