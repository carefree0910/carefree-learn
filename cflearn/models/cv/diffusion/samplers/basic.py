import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from .schema import ISampler
from .schema import IDiffusion
from .schema import DDPMQSampler
from ..utils import cond_type
from ..utils import extract_to
from ..utils import get_timesteps


@ISampler.register("basic")
class BasicSampler(ISampler):
    def __init__(
        self,
        model: IDiffusion,
        *,
        temperature: float = 1.0,
        default_steps: Optional[int] = None,
    ):
        super().__init__(model)
        if default_steps is None:
            default_steps = model.t
        self.temperature = temperature
        self.default_steps = default_steps

    @property
    def q_sampler(self) -> DDPMQSampler:
        return self.model.q_sampler

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(temperature=self.temperature)

    def sample_step(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        *,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Tensor:
        shape = image.shape
        num_dim = len(shape)
        ts = get_timesteps(total_step - step - 1, shape[0], image.device)
        net = self.model.denoise(image, ts, cond, step, total_step)
        parameterization = self.model.parameterization
        if parameterization == "eps":
            coef1 = extract_to(self.model.posterior_coef1, ts, num_dim)
            coef2 = extract_to(self.model.posterior_coef2, ts, num_dim)
            mean = coef1 * (image - coef2 * net)
        elif parameterization == "x0":
            mean = net
        else:
            msg = f"unrecognized parameterization '{parameterization}' occurred"
            raise NotImplementedError(msg)
        noise = torch.randn_like(image) * temperature
        noise_mask_shape = shape[0], *((1,) * (num_dim - 1))
        noise_mask = (1.0 - (ts == 0).float()).view(noise_mask_shape)
        log_var = extract_to(self.model.posterior_log_variance_clipped, ts, num_dim)
        return mean + noise_mask * (0.5 * log_var).exp() * noise


__all__ = [
    "BasicSampler",
]
