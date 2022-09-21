from torch import Tensor
from typing import Any
from typing import List
from typing import Optional

from .ddim import DDIMMixin
from .ddim import IGetEPSPred
from .ddim import IGetDenoised
from .protocol import ISampler


@ISampler.register("plms")
class PLMSSampler(DDIMMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        step: int,
        total_step: int,
        get_eps_pred: IGetEPSPred,
        get_denoised: IGetDenoised,
        *,
        eta: float,
        discretize: str,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        temperature: float,
        noise_dropout: float,
        quantize_denoised: bool,
        **kwargs: Any,
    ) -> Tensor:
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
