import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.array import save_images

from ....data import predict_tensor_data
from ....data import TensorInferenceData
from ....pipeline import DLPipeline
from ....constants import INPUT_KEY
from ....misc.toolkit import eval_context
from ....models.cv.diffusion import LDM
from ....models.cv.diffusion import DDPM
from ....models.cv.diffusion import ISampler
from ....models.cv.diffusion import DDIMSampler
from ....models.cv.diffusion import PLMSSampler
from ....models.cv.ae.common import IAutoEncoder
from ....models.cv.diffusion.utils import q_sample
from ....models.cv.diffusion.utils import get_timesteps

try:
    from cfcv.misc.toolkit import to_rgb
except:
    to_rgb = None


def is_ddim(sampler: ISampler) -> bool:
    return isinstance(sampler, (DDIMSampler, PLMSSampler))


def get_normalized(path: str, max_wh: int, *, to_gray: bool = False) -> np.ndarray:
    if to_rgb is None:
        raise ValueError("`carefree-cv` is needed for `DiffusionAPI`")
    image = Image.open(path)
    image = image.convert("L") if to_gray else to_rgb(image)
    original_w, original_h = image.size
    max_original_wh = max(original_w, original_h)
    if max_original_wh <= max_wh:
        w, h = original_w, original_h
    else:
        wh_ratio = original_w / original_h
        if wh_ratio >= 1:
            w = max_wh
            h = round(w / wh_ratio)
        else:
            h = max_wh
            w = round(h * wh_ratio)
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    if to_gray:
        image = image[None, None]
    else:
        image = image[None].transpose(0, 3, 1, 2)
    return image


class DiffusionAPI:
    m: DDPM
    cond_model: Optional[nn.Module]
    first_stage: Optional[IAutoEncoder]

    def __init__(self, m: DDPM):
        self.m = m
        self.cond_type = m.condition_type
        self.cond_model = m.condition_model
        m.condition_model = nn.Identity()
        if is_ddim(m.sampler):
            if self.cond_model is not None and m.sampler.unconditional_cond is not None:
                uncond = self.cond_model(m.sampler.unconditional_cond)
                m.sampler.unconditional_cond = uncond.to(m.device)
        if not isinstance(m, LDM):
            self.first_stage = None
        else:
            self.first_stage = m.first_stage.core

    def switch_sampler(
        self,
        sampler: str,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        sampler_ins = self.m.make_sampler(sampler, sampler_config)
        if is_ddim(self.m.sampler) and is_ddim(sampler_ins):
            sampler_ins.unconditional_cond = self.m.sampler.unconditional_cond  # type: ignore

    def sample(
        self,
        num_samples: int,
        export_path: Optional[str] = None,
        *,
        z: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        callback: Optional[Callable[[Tensor], Tensor]] = None,
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if cond is not None:
            if self.cond_type != "concat" and self.cond_model is not None:
                data = TensorInferenceData(cond, batch_size=batch_size)
                cond = predict_tensor_data(self.cond_model, data)
        if cond is not None and num_samples != len(cond):
            raise ValueError(
                f"`num_samples` ({num_samples}) should be identical with "
                f"the number of `cond` ({len(cond)})"
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
        if size is None:
            size = self.m.img_size, self.m.img_size
        else:
            if self.first_stage is None:
                factor = 1
            else:
                factor = self.first_stage.img_size // self.m.img_size
            size = tuple(map(lambda n: round(n / factor), size))  # type: ignore
        with eval_context(self.m):
            for batch in iterator:
                i_kw = shallow_copy_dict(kw)
                i_cond = batch[INPUT_KEY].to(self.m.device)
                if z is not None:
                    i_z = z.repeat_interleave(len(i_cond), dim=0)
                else:
                    i_z_shape = len(i_cond), self.m.in_channels, *size[::-1]
                    i_z = torch.randn(i_z_shape, device=self.m.device)
                if unconditional:
                    i_cond = None
                i_sampled = self.m.decode(i_z, cond=i_cond, **i_kw)
                sampled.append(i_sampled.cpu())
        concat = torch.cat(sampled, dim=0)
        if clip_output:
            concat = torch.clip(concat, -1.0, 1.0)
        if callback is not None:
            concat = callback(concat)
        if export_path is not None:
            save_images(concat, export_path)
        return concat

    def img2img(
        self,
        img_path: str,
        export_path: Optional[str] = None,
        *,
        max_wh: int = 1024,
        fidelity: float = 0.2,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        img = get_normalized(img_path, max_wh)
        img = 2.0 * img - 1.0
        z = self._get_z(img)
        return self._img2img(
            z,
            export_path,
            fidelity=fidelity,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )

    def inpainting(
        self,
        img_path: str,
        mask_path: str,
        export_path: Optional[str] = None,
        *,
        max_wh: int = 1024,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # callback
        def callback(out: Tensor) -> Tensor:
            final = torch.from_numpy(remained_image.copy())
            final += 0.5 * (1.0 + out) * (1.0 - remained_mask)
            return 2.0 * final - 1.0

        # handle mask stuffs
        image = get_normalized(img_path, max_wh)
        mask = get_normalized(mask_path, max_wh, to_gray=True)
        bool_mask = mask >= 0.5
        remained_mask = (~bool_mask).astype(np.float32)
        remained_image = remained_mask * image
        # construct condition tensor
        remained_cond = self._get_z(2.0 * remained_image - 1.0)
        mask_cond = torch.where(torch.from_numpy(bool_mask), 1.0, -1.0)
        mask_cond = mask_cond.to(torch.float32).to(self.m.device)
        mask_cond = F.interpolate(mask_cond, size=remained_cond.shape[-2:])
        cond = torch.cat([remained_cond, mask_cond], dim=1)
        # sampling
        z = torch.randn_like(remained_cond)
        return self.sample(
            1,
            export_path,
            z=z,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )

    @classmethod
    def from_pipeline(cls, m: DLPipeline) -> "DiffusionAPI":
        return cls(m.model.core)

    def _get_z(self, img: np.ndarray) -> Tensor:
        z = torch.from_numpy(img).to(self.m.device)
        z = self.m._preprocess(z)
        return z

    def _img2img(
        self,
        z: Tensor,
        export_path: Optional[str] = None,
        *,
        fidelity: float = 0.2,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # perturb z
        sampler = self.m.sampler
        if num_steps is None:
            num_steps = sampler.default_steps
        t = round((1.0 - fidelity) * num_steps)
        ts = get_timesteps(t, 1, z.device)
        if not isinstance(sampler, (DDIMSampler, PLMSSampler)):
            z = self.m._q_sample(z, ts)
        else:
            kw = shallow_copy_dict(sampler.sample_kwargs)
            kw["total_step"] = num_steps
            safe_execute(sampler._reset_buffers, kw)
            z = q_sample(
                z,
                ts,
                torch.sqrt(sampler.ddim_alphas),
                sampler.ddim_sqrt_one_minus_alphas,
            )
        kwargs["start_step"] = num_steps - t
        # sampling
        return self.sample(
            1,
            export_path,
            z=z,
            cond=cond,
            num_steps=num_steps,
            clip_output=clip_output,
            verbose=verbose,
            **kwargs,
        )


__all__ = [
    "DiffusionAPI",
]
