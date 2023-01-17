import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
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
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cftool.array import arr_type
from cftool.array import save_images
from cftool.types import tensor_dict_type

from .common import read_image
from .common import restrict_wh
from .common import get_suitable_size
from .common import APIMixin
from .common import ReadImageResponse
from ....zoo import DLZoo
from ....data import predict_tensor_data
from ....data import TensorInferenceData
from ....pipeline import DLPipeline
from ....constants import INPUT_KEY
from ....misc.toolkit import slerp
from ....misc.toolkit import new_seed
from ....misc.toolkit import eval_context
from ....misc.toolkit import seed_everything
from ....modules.blocks import Conv2d
from ....models.cv.diffusion import LDM
from ....models.cv.diffusion import DDPM
from ....models.cv.diffusion import ISampler
from ....models.cv.ae.common import IAutoEncoder
from ....models.cv.diffusion.utils import get_timesteps
from ....models.cv.diffusion.utils import CONCAT_KEY
from ....models.cv.diffusion.utils import CONCAT_TYPE
from ....models.cv.diffusion.utils import HYBRID_TYPE
from ....models.cv.diffusion.utils import CROSS_ATTN_KEY
from ....models.cv.diffusion.cond_models import CLIPTextConditionModel
from ....models.cv.diffusion.samplers.ddim import DDIMMixin
from ....models.cv.diffusion.samplers.solver import DPMSolver
from ....models.cv.diffusion.samplers.k_samplers import KSamplerMixin


class switch_sampler_context:
    def __init__(self, api: "DiffusionAPI", sampler: Optional[str]):
        self.api = api
        self.m_sampler = api.sampler.__identifier__
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


class SizeInfo(NamedTuple):
    factor: int
    opt_size: int


class MaskedCond(NamedTuple):
    image_res: ReadImageResponse
    mask_res: ReadImageResponse
    mask: np.ndarray
    remained_image: np.ndarray
    remained_mask: np.ndarray
    mask_cond: Tensor
    remained_image_cond: Tensor


class DiffusionAPI(APIMixin):
    m: DDPM
    sampler: ISampler
    cond_model: Optional[nn.Module]
    first_stage: Optional[IAutoEncoder]
    latest_seed: int
    latest_variation_seed: Optional[int]

    def __init__(
        self,
        m: DDPM,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
        clip_skip: int = 0,
    ):
        super().__init__(m, device, use_amp=use_amp, use_half=use_half)
        self.sampler = m.sampler
        self.cond_type = m.condition_type
        self.clip_skip = clip_skip
        # extracted the condition model so we can pre-calculate the conditions
        self.cond_model = m.condition_model
        if self.cond_model is not None:
            self.cond_model.eval()
        m.condition_model = nn.Identity()
        # pre-calculate unconditional_cond if needed
        self._original_raw_uncond = getattr(m.sampler, "unconditional_cond", None)
        self._uncond_cache: tensor_dict_type = {}
        self._update_sampler_uncond(clip_skip)
        # extract first stage
        if not isinstance(m, LDM):
            self.first_stage = None
        else:
            self.first_stage = m.first_stage.core

    def to(
        self,
        device: torch.device,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> None:
        super().to(device, use_amp=use_amp, use_half=use_half)
        unconditional_cond = getattr(self.sampler, "unconditional_cond", None)
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
            self.sampler.unconditional_cond = unconditional_cond.to(device)
        for k, v in self._uncond_cache.items():
            self._uncond_cache[k] = v.to(device)

    @property
    def size_info(self) -> SizeInfo:
        opt_size = self.m.img_size
        if self.first_stage is None:
            factor = 1
        else:
            factor = self.first_stage.img_size // opt_size
        return SizeInfo(factor, opt_size)

    def get_cond(self, cond: Any) -> Tensor:
        if self.cond_model is None:
            msg = "should not call `get_cond` when `cond_model` is not available"
            raise ValueError(msg)
        with torch.no_grad():
            with self.amp_context:
                return self.cond_model(cond)

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
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
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
            if self.cond_type != CONCAT_TYPE and self.cond_model is not None:
                data = TensorInferenceData(cond, batch_size=batch_size)
                cond = predict_tensor_data(self.cond_model, data)
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
        iterator = TensorInferenceData(cond, batch_size=batch_size).initialize()[0]
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
        with eval_context(self.m):
            with self.amp_context:
                for batch in iterator:
                    i_kw = shallow_copy_dict(kw)
                    i_cond = batch[INPUT_KEY].to(self.device)
                    repeat = lambda t: t.repeat_interleave(len(i_cond), dim=0)
                    if z is not None:
                        i_z = repeat(z)
                    else:
                        in_channels = self.m.in_channels
                        if self.cond_type == CONCAT_TYPE:
                            in_channels -= cond.shape[1]
                        elif cond_concat is not None:
                            in_channels -= cond_concat.shape[1]
                        i_z_shape = len(i_cond), in_channels, *size[::-1]
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
                                len(i_cond),
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
                        if self.cond_type != HYBRID_TYPE:
                            raise ValueError(
                                f"condition type should be `{HYBRID_TYPE}` when "
                                f"`cond_concat` is provided"
                            )
                        i_cond = {
                            CROSS_ATTN_KEY: i_cond,
                            CONCAT_KEY: cond_concat,
                        }
                    with switch_sampler_context(self, i_kw.get("sampler")):
                        i_sampled = self.m.decode(i_z, cond=i_cond, **i_kw)
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
        return concat

    def _update_clip_skip(self, clip_skip: int) -> None:
        if isinstance(self.cond_model, CLIPTextConditionModel):
            self.cond_model.clip_skip = clip_skip

    def _update_sampler_uncond(self, clip_skip: int) -> None:
        self._update_clip_skip(clip_skip)
        if self.cond_model is not None and self._original_raw_uncond is not None:
            cache = self._uncond_cache.get(clip_skip)
            if cache is not None:
                uncond = cache
            else:
                uncond = self.get_cond(self._original_raw_uncond)
                self._uncond_cache[clip_skip] = uncond
            self.m.sampler.unconditional_cond = uncond.to(self.device)

    @staticmethod
    def _txt_cond(t: Union[str, List[str]], n: Optional[int]) -> Tuple[List[str], int]:
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

    @staticmethod
    def _get_size(
        size: Optional[Tuple[int, int]],
        anchor: int,
        max_wh: int,
    ) -> Optional[Tuple[int, int]]:
        if size is None:
            return None
        new_size = restrict_wh(*size, max_wh)
        return tuple(map(get_suitable_size, new_size, (anchor, anchor)))  # type: ignore

    def _get_masked_cond(
        self,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        max_wh: int,
        anchor: int,
        mask_image_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mask_cond_fn: Callable[[np.ndarray], Tensor],
    ) -> MaskedCond:
        # handle mask stuffs
        image_res = read_image(image, max_wh, anchor=anchor)
        mask_res = read_image(mask, max_wh, anchor=anchor, to_mask=True)
        mask = mask_res.image
        bool_mask = mask >= 0.5
        remained_mask = (~bool_mask).astype(np.float16 if self.use_half else np.float32)
        remained_image = mask_image_fn(remained_mask, image_res.image)
        # construct condition tensor
        remained_cond = self._get_z(remained_image)
        latent_shape = remained_cond.shape[-2:]
        mask_cond = mask_cond_fn(bool_mask)
        mask_cond = mask_cond.to(torch.float16 if self.use_half else torch.float32)
        mask_cond = mask_cond.to(self.device)
        mask_cond = F.interpolate(mask_cond, size=latent_shape)
        return MaskedCond(
            image_res,
            mask_res,
            mask,
            remained_image,
            remained_mask,
            mask_cond,
            remained_cond,
        )

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
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        txt, num_samples = self._txt_cond(txt, num_samples)
        new_size = self._get_size(size, anchor, max_wh)
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
        txt: str,
        image: Union[str, Image.Image],
        mask: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 64,
        max_wh: int = 512,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        keep_original: bool = False,
        use_raw_inpainting: bool = False,
        raw_inpainting_fidelity: float = 0.2,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if use_raw_inpainting:
            image_res = read_image(image, max_wh, anchor=anchor)
            mask = read_image(mask, max_wh, anchor=anchor, to_mask=True).image
            z = z_ref = self._get_z(image_res.image)
            z_ref_mask = F.interpolate(
                torch.from_numpy(mask).to(z_ref),
                z_ref.shape[-2:],
                mode="bicubic",
            )
            return self._img2img(
                z,
                export_path,
                z_ref=z_ref,
                z_ref_mask=z_ref_mask,
                fidelity=raw_inpainting_fidelity,
                original_size=image_res.original_size,
                alpha=None,
                cond=[txt],
                num_steps=num_steps,
                clip_output=clip_output,
                verbose=verbose,
                **kwargs,
            )
        txt_list, num_samples = self._txt_cond(txt, 1)
        res = self._get_masked_cond(
            image,
            mask,
            max_wh,
            anchor,
            lambda remained_mask, img: np.where(remained_mask, img, 0.5),
            lambda bool_mask: torch.from_numpy(bool_mask),
        )
        # sampling
        size = tuple(
            map(
                lambda n: n * self.size_info.factor,
                res.remained_image_cond.shape[-2:][::-1],
            )
        )
        sampled = self.sample(
            num_samples,
            export_path,
            size=size,  # type: ignore
            original_size=res.image_res.original_size,
            alpha=res.image_res.alpha,
            cond=txt_list,
            cond_concat=torch.cat([res.mask_cond, res.remained_image_cond], dim=1),
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )
        if keep_original:
            original = np.array(res.image_res.original).astype(np.float32) / 127.5 - 1.0
            original = original.transpose([2, 0, 1])[None]
            mask_ = read_image(res.mask_res.original, None, anchor=None, to_mask=True)
            remained_mask = ~(mask_.image >= 0.5)
            sampled = np.where(remained_mask, original, sampled.numpy())
            sampled = torch.from_numpy(sampled)
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
        size = tuple(
            map(
                lambda n: n * self.size_info.factor,
                res.remained_image_cond.shape[-2:][::-1],
            )
        )
        # refine with img2img
        if refine_fidelity is not None:
            z = self._get_z(res.image_res.image)
            return self._img2img(
                z,
                export_path,
                fidelity=refine_fidelity,
                original_size=res.image_res.original_size,
                alpha=res.image_res.alpha if alpha is None else alpha,
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
            original_size=res.image_res.original_size,
            alpha=res.image_res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    def sr(
        self,
        image: Union[str, Image.Image],
        export_path: Optional[str] = None,
        *,
        anchor: int = 8,
        max_wh: int = 512,
        alpha: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if not isinstance(self.m, LDM):
            raise ValueError("`sr` is now only available for `LDM` models")
        factor = 2 ** (len(self.m.first_stage.core.channel_multipliers) - 1)
        res = read_image(image, round(max_wh / factor), anchor=anchor)
        wh_ratio = res.original_size[0] / res.original_size[1]
        zh, zw = res.image.shape[-2:]
        sr_size = (zw, zw / wh_ratio) if zw > zh else (zh * wh_ratio, zh)
        sr_size = tuple(map(lambda n: round(factor * n), sr_size))  # type: ignore
        cond = torch.from_numpy(2.0 * res.image - 1.0).to(self.device)
        z = torch.randn_like(cond)
        return self.sample(
            1,
            export_path,
            z=z,
            original_size=sr_size,
            alpha=res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
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
        cond = cond.half() if self.use_half else cond.float()
        cond = cond.permute(0, 3, 1, 2).contiguous()
        cond = self.get_cond(cond)
        z = torch.randn_like(cond)
        return self.sample(
            1,
            export_path,
            z=z,
            original_size=res.original_size,
            alpha=res.alpha if alpha is None else alpha,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def load_context(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, api: DiffusionAPI):
                self.api = api
                self.m_cond = api.m.condition_model
                api.m.condition_model = api.cond_model

            def __enter__(self) -> nn.Module:
                class Wrapper(nn.Module):
                    def __init__(self, m: nn.Module) -> None:
                        super().__init__()
                        self.core = m

                return Wrapper(self.api.m)

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.api.m.condition_model = self.m_cond

        return _(self)

    @classmethod
    def from_sd(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        return cls.from_pipeline(ldm_sd(), device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_version(
        cls,
        version: str,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_sd_version(version)
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_anime(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_sd_anime()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_inpainting(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_sd_inpainting()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sd_v2(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        return cls.from_pipeline(
            ldm_sd_v2(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=1,
        )

    @classmethod
    def from_sd_v2_base(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        return cls.from_pipeline(
            ldm_sd_v2_base(),
            device,
            use_amp=use_amp,
            use_half=use_half,
            clip_skip=1,
        )

    @classmethod
    def from_celeba_hq(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_celeba_hq()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_inpainting(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_inpainting()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_sr(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        return cls.from_pipeline(ldm_sr(), device, use_amp=use_amp, use_half=use_half)

    @classmethod
    def from_semantic(
        cls,
        device: Optional[str] = None,
        *,
        use_amp: bool = False,
        use_half: bool = False,
    ) -> "DiffusionAPI":
        m = ldm_semantic()
        return cls.from_pipeline(m, device, use_amp=use_amp, use_half=use_half)

    def _get_z(self, img: arr_type) -> Tensor:
        img = 2.0 * img - 1.0
        z = img if isinstance(img, Tensor) else torch.from_numpy(img)
        if self.use_half:
            z = z.half()
        z = z.to(self.device)
        z = self.m._preprocess(z, deterministic=True)
        return z

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

    def _img2img(
        self,
        z: Tensor,
        export_path: Optional[str] = None,
        *,
        seed: Optional[int] = None,
        # each variation contains (seed, weight)
        variations: Optional[List[Tuple[int, float]]] = None,
        variation_seed: Optional[int] = None,
        variation_strength: Optional[float] = None,
        z_ref: Optional[Tensor] = None,
        z_ref_mask: Optional[Tensor] = None,
        fidelity: float = 0.2,
        original_size: Optional[Tuple[int, int]] = None,
        alpha: Optional[np.ndarray] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        with switch_sampler_context(self, kwargs.get("sampler")):
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
            kwargs["start_step"] = num_steps - t
            return self.sample(
                1,
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


def _ldm(
    model: str,
    latent_size: int,
    latent_in_channels: int,
    latent_out_channels: int,
    **kwargs: Any,
) -> DLPipeline:
    kwargs["img_size"] = latent_size
    kwargs["in_channels"] = latent_in_channels
    kwargs["out_channels"] = latent_out_channels
    model_config = kwargs.setdefault("model_config", {})
    first_stage_kw = model_config.setdefault("first_stage_config", {})
    first_stage_kw.setdefault("report", False)
    first_stage_kw.setdefault("pretrained", True)
    first_stage_model_config = first_stage_kw.setdefault("model_config", {})
    use_loss = first_stage_model_config.setdefault("use_loss", False)
    if not use_loss:

        def state_callback(states: tensor_dict_type) -> tensor_dict_type:
            for key in list(states.keys()):
                if key.startswith("core.loss"):
                    states.pop(key)
            return states

        first_stage_kw["pretrained_state_callback"] = state_callback
    return DLZoo.load_pipeline(model, **kwargs)


def ldm(
    latent_size: int = 32,
    latent_in_channels: int = 4,
    latent_out_channels: int = 4,
    **kwargs: Any,
) -> DLPipeline:
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
) -> DLPipeline:
    return _ldm(
        "diffusion/ldm.vq",
        latent_size,
        latent_in_channels,
        latent_out_channels,
        **kwargs,
    )


def ldm_sd(pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    return _ldm("diffusion/ldm.sd", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_sd_version(version: str, pretrained: bool = True) -> DLPipeline:
    return ldm_sd(pretrained, download_name=f"ldm_sd_{version}")


def ldm_sd_anime(pretrained: bool = True) -> DLPipeline:
    return ldm_sd_version("anime_nai", pretrained)


def ldm_sd_inpainting(pretrained: bool = True, **kw: Any) -> DLPipeline:
    return _ldm("diffusion/ldm.sd_inpainting", 64, 9, 4, pretrained=pretrained, **kw)


def ldm_sd_v2(pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    return _ldm("diffusion/ldm.sd_v2", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_sd_v2_base(pretrained: bool = True, **kwargs: Any) -> DLPipeline:
    return _ldm("diffusion/ldm.sd_v2_base", 64, 4, 4, pretrained=pretrained, **kwargs)


def ldm_celeba_hq(pretrained: bool = True) -> DLPipeline:
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


def ldm_inpainting(pretrained: bool = True) -> DLPipeline:
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


def ldm_sr(pretrained: bool = True) -> DLPipeline:
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


def ldm_semantic(pretrained: bool = True) -> DLPipeline:
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


__all__ = [
    "DiffusionAPI",
]
