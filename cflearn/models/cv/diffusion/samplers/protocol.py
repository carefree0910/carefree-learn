import torch

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
from cftool.types import tensor_dict_type


samplers: Dict[str, Type["ISampler"]] = {}


class Denoise(Protocol):
    def __call__(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond_kw: tensor_dict_type,
    ) -> Tensor:
        pass


class IDiffusion:
    _get_cond: Callable
    _get_input: Callable

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
        cond_kw: tensor_dict_type,
        step: int,
        total_step: int,
        **kwargs: Any,
    ) -> Tensor:
        pass

    def sample(
        self,
        z: Tensor,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if cond is not None and self.model.condition_model is not None:
            cond = self.model._get_cond(cond)
        image, cond_kw = self.model._get_input(z, cond, in_decode=True)
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
        for step in iterator:
            kw = shallow_copy_dict(self.sample_kwargs)
            update_dict(shallow_copy_dict(kwargs), kw)
            image = self.sample_step(image, cond_kw, step, num_steps, **kw)
        return image


__all__ = [
    "ISampler",
]
