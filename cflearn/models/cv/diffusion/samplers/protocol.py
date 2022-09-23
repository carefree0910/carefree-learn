import math
import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from tqdm import tqdm
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from typing import Optional
from typing import Protocol
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister

from ..utils import q_sample
from ..utils import get_timesteps


samplers: Dict[str, Type["ISampler"]] = {}


class Denoise(Protocol):
    def __call__(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond: Optional[Tensor],
    ) -> Tensor:
        pass


class IDiffusion:
    _get_cond: Callable
    _q_sample: Callable

    denoise: Denoise
    condition_model: Optional[nn.Module]
    first_stage: nn.Module
    device: torch.device

    t: int
    parameterization: str
    posterior_coef1: Tensor
    posterior_coef2: Tensor
    posterior_log_variance_clipped: Tensor

    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor


class ISampler(WithRegister, metaclass=ABCMeta):
    d = samplers

    default_steps: int

    def __init__(self, model: IDiffusion):
        self.model = model

    @property
    @abstractmethod
    def sample_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def sample_step(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        step: int,
        total_step: int,
        **kwargs: Any,
    ) -> Tensor:
        pass

    @abstractmethod
    def q_sample(self, net: Tensor, timesteps: Tensor) -> Tensor:
        pass

    def sample(
        self,
        z: Tensor,
        *,
        ref: Optional[Tensor] = None,
        ref_mask: Optional[Tensor] = None,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # setup
        if num_steps is None:
            num_steps = getattr(self, "default_steps", self.model.t)
            assert isinstance(num_steps, int)
        if start_step is None:
            start_step = 0
        iterator = list(range(start_step, num_steps))
        if verbose:
            iterator = tqdm(iterator, desc=f"sampling ({self.__identifier__})")
        # execute
        image = z
        if cond is not None and self.model.condition_model is not None:
            cond = self.model._get_cond(cond)
        for step in iterator:
            kw = shallow_copy_dict(self.sample_kwargs)
            update_dict(shallow_copy_dict(kwargs), kw)
            image = self.sample_step(image, cond, step, num_steps, **kw)
            if ref is not None and ref_mask is not None:
                ref_ts = get_timesteps(num_steps - step - 1, ref.shape[0], z.device)
                ref_noisy = self.q_sample(ref, ref_ts)
                image = ref_noisy * ref_mask + image * (1.0 - ref_mask)
        return image


class QSampleMixin:
    model: IDiffusion

    def q_sample(self, net: Tensor, timesteps: Tensor) -> Tensor:
        return q_sample(
            net,
            timesteps,
            torch.sqrt(self.q_alphas),
            self.q_sqrt_one_minus_alphas,
        )

    def _reset_q_buffers(self, discretize: str, total_step: int) -> None:
        if discretize == "uniform":
            span = self.model.t // total_step
            q_timesteps = np.array(list(range(0, self.model.t, span)))
        elif discretize == "quad":
            end = math.sqrt(self.model.t * 0.8)
            q_timesteps = (np.linspace(0, end, total_step) ** 2).astype(int)
        else:
            raise ValueError(f"unrecognized discretize method '{discretize}' occurred")
        q_timesteps += 1
        alphas = self.model.alphas_cumprod
        self.q_alphas = alphas[q_timesteps]
        self.q_timesteps = q_timesteps
        self.q_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.q_alphas)


class UncondSamplerMixin:
    model: IDiffusion
    uncond: Optional[Tensor]
    unconditional_cond: Optional[Any]
    uncond_guidance_scale: float

    def _reset_uncond_buffers(
        self,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        if unconditional_cond is None or self.model.condition_model is None:
            self.uncond = None
            self.uncond_guidance_scale = 0.0
        else:
            self.uncond = self.model._get_cond(unconditional_cond)
            self.uncond_guidance_scale = unconditional_guidance_scale

    def _uncond_denoise(
        self,
        image: Tensor,
        ts: Tensor,
        cond: Optional[Tensor],
    ) -> Tensor:
        if cond is None or self.uncond is None:
            return self.model.denoise(image, ts, cond)
        uncond = self.uncond.repeat_interleave(cond.shape[0], dim=0)
        cond2 = torch.cat([uncond, cond])
        image2 = torch.cat([image, image])
        ts2 = torch.cat([ts, ts])
        eps_uncond, eps = self.model.denoise(image2, ts2, cond2).chunk(2)
        return eps_uncond + self.uncond_guidance_scale * (eps - eps_uncond)


__all__ = [
    "ISampler",
]
