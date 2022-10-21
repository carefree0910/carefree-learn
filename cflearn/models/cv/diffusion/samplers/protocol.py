import torch

import torch.nn as nn

from abc import abstractmethod
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

from ..utils import cond_type
from ..utils import extract_to
from ..utils import get_timesteps
from ..utils import CONCAT_KEY


samplers: Dict[str, Type["ISampler"]] = {}


class Denoise(Protocol):
    def __call__(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond: Optional[cond_type],
    ) -> Tensor:
        pass


class IDiffusion:
    _get_cond: Callable

    denoise: Denoise
    q_sampler: "DDPMQSampler"
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


class IQSampler:
    def __init__(self, model: IDiffusion):
        self.model = model

    @abstractmethod
    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def reset_buffers(self, **kwargs: Any) -> None:
        pass


class DDPMQSampler(IQSampler):
    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        num_dim = len(net.shape)
        w_net = extract_to(self.sqrt_alphas, timesteps, num_dim)
        w_noise = extract_to(self.sqrt_one_minus_alphas, timesteps, num_dim)
        if noise is None:
            noise = torch.randn_like(net)
        net = w_net * net + w_noise * noise
        return net

    def reset_buffers(self, sqrt_alpha: Tensor, sqrt_one_minus_alpha: Tensor) -> None:  # type: ignore
        self.sqrt_alphas = sqrt_alpha
        self.sqrt_one_minus_alphas = sqrt_one_minus_alpha


class ISampler(WithRegister):
    d = samplers

    default_steps: int

    def __init__(self, model: IDiffusion):
        self.model = model

    @property
    @abstractmethod
    def q_sampler(self) -> IQSampler:
        pass

    @property
    @abstractmethod
    def sample_kwargs(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def sample_step(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        **kwargs: Any,
    ) -> Tensor:
        pass

    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        return self.q_sampler.q_sample(net, timesteps, noise)

    def sample(
        self,
        z: Tensor,
        *,
        ref: Optional[Tensor] = None,
        ref_mask: Optional[Tensor] = None,
        ref_noise: Optional[Tensor] = None,
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
            if ref is not None and ref_mask is not None and ref_noise is not None:
                ref_ts = get_timesteps(num_steps - step - 1, ref.shape[0], z.device)
                ref_noisy = self.q_sample(ref, ref_ts, ref_noise)
                image = ref_noisy * ref_mask + image * (1.0 - ref_mask)
        return image


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
        cond: Optional[cond_type],
    ) -> Tensor:
        if cond is None or self.uncond is None:
            return self.model.denoise(image, ts, cond)
        uncond = self.uncond.repeat_interleave(image.shape[0], dim=0)
        if not isinstance(cond, dict):
            cond2 = torch.cat([uncond, cond])
        else:
            cond2 = shallow_copy_dict(cond)
            for k, v in cond2.items():
                if k != CONCAT_KEY:
                    cond2[k] = torch.cat([uncond, v])
        image2 = torch.cat([image, image])
        ts2 = torch.cat([ts, ts])
        eps_uncond, eps = self.model.denoise(image2, ts2, cond2).chunk(2)
        return eps_uncond + self.uncond_guidance_scale * (eps - eps_uncond)


__all__ = [
    "ISampler",
    "IQSampler",
    "DDPMQSampler",
]
