import torch

import torch.nn as nn

from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
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


def is_ddim(sampler: ISampler) -> bool:
    return isinstance(sampler, (DDIMSampler, PLMSSampler))


class DiffusionAPI:
    m: DDPM
    cond_model: Optional[nn.Module]
    first_stage: Optional[IAutoEncoder]

    def __init__(self, m: DDPM):
        self.m = m
        self.cond_model = m.condition_model
        m.condition_model = nn.Identity()
        if is_ddim(m.sampler):
            if m.sampler.unconditional_cond is not None:
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
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if cond is not None and self.cond_model is not None:
            data = TensorInferenceData(cond, batch_size=batch_size)
            cond = predict_tensor_data(self.cond_model, data)
        if cond is not None and num_samples != len(cond):
            raise ValueError(
                f"`num_samples` ({num_samples}) should be identical with "
                f"the number of `cond` ({len(cond)})"
            )
        iterator = TensorInferenceData(cond, batch_size=batch_size).initialize()[0]
        num_iter = len(iterator)
        if verbose and num_iter > 1:
            iterator = tqdm(iterator, desc="iter", total=num_iter)
        sampled = []
        kw = dict(num_steps=num_steps, clip_output=clip_output, verbose=verbose)
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
                i_sampled = self.m.decode(i_z, cond=i_cond, **i_kw)
                sampled.append(i_sampled.cpu())
        concat = torch.cat(sampled, dim=0)
        if export_path is not None:
            save_images(concat, export_path)
        return concat

    @classmethod
    def from_pipeline(cls, m: DLPipeline) -> "DiffusionAPI":
        return cls(m.model.core)


__all__ = [
    "DiffusionAPI",
]
