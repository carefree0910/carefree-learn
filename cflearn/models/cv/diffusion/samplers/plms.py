from torch import Tensor
from typing import Any
from typing import List
from typing import Optional
from cftool.types import tensor_dict_type

from .ddim import DDIMMixin
from .protocol import ISampler


@ISampler.register("plms")
class PLMSSampler(DDIMMixin):
    def sample_step(
        self,
        image: Tensor,
        cond_kw: tensor_dict_type,
        step: int,
        total_step: int,
        *,
        eta: float = 0.0,
        discretize: str = "uniform",
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        temperature: float = 1.0,
        noise_dropout: float = 1.0,
        quantize_denoised: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if step == 0:
            self._reset_buffers(
                eta,
                discretize,
                total_step,
                unconditional_cond,
                unconditional_guidance_scale,
            )
        self._register_temp_buffers(image, step, total_step)

        get_eps_pred = lambda img, ts: self._denoise(img, ts, cond_kw)
        get_denoised = lambda eps: self._get_denoised_and_pred_x0(
            eps,
            image,
            quantize_denoised,
            temperature,
            noise_dropout,
        )[0]

        eps_pred = get_eps_pred(image, self._ts)

        if len(self.old_eps) == 0:
            denoised = get_denoised(eps_pred)
            eps_pred_next = get_eps_pred(denoised, self._ts_next)
            eps_prime = 0.5 * (eps_pred + eps_pred_next)
        elif len(self.old_eps) == 1:
            eps_prime = 0.5 * (3.0 * eps_pred - self.old_eps[-1])
        elif len(self.old_eps) == 2:
            eps_prime = (
                23.0 * eps_pred - 16.0 * self.old_eps[-1] + 5.0 * self.old_eps[-2]
            ) / 12.0
        elif len(self.old_eps) >= 3:
            eps_prime = (
                55.0 * eps_pred
                - 59.0 * self.old_eps[-1]
                + 37.0 * self.old_eps[-2]
                - 9.0 * self.old_eps[-3]
            ) / 24.0
        else:
            raise ValueError("length of `old_eps` should be in {0, 1, 2, 3}")

        self.old_eps.append(eps_pred)
        if len(self.old_eps) >= 4:
            self.old_eps.pop(0)

        denoised = get_denoised(eps_prime)
        return denoised

    def _reset_buffers(
        self,
        eta: float,
        discretize: str,
        total_step: int,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        super()._reset_buffers(
            eta,
            discretize,
            total_step,
            unconditional_cond,
            unconditional_guidance_scale,
        )
        self.old_eps: List[Tensor] = []


__all__ = [
    "PLMSSampler",
]
