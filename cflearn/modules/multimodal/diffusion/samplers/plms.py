from torch import Tensor
from typing import Any
from typing import List
from typing import Optional

from .ddim import DDIMMixin
from .ddim import IGetModelOutput
from .ddim import IGetDenoised
from .schema import ISampler
from ..utils import cond_type


@ISampler.register("plms")
class PLMSSampler(DDIMMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_model_output: IGetModelOutput,
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
        model_output = get_model_output(image, self._ts)

        if len(self.old_outputs) == 0:
            denoised = get_denoised(model_output, self._ts_next)
            model_output_next = get_model_output(denoised, self._ts_next)
            model_output_prime = 0.5 * (model_output + model_output_next)
        elif len(self.old_outputs) == 1:
            model_output_prime = 0.5 * (3.0 * model_output - self.old_outputs[-1])
        elif len(self.old_outputs) == 2:
            model_output_prime = (
                23.0 * model_output
                - 16.0 * self.old_outputs[-1]
                + 5.0 * self.old_outputs[-2]
            ) / 12.0
        elif len(self.old_outputs) >= 3:
            model_output_prime = (
                55.0 * model_output
                - 59.0 * self.old_outputs[-1]
                + 37.0 * self.old_outputs[-2]
                - 9.0 * self.old_outputs[-3]
            ) / 24.0
        else:
            raise ValueError("length of `old_eps` should be in {0, 1, 2, 3}")

        self.old_outputs.append(model_output)
        if len(self.old_outputs) >= 4:
            self.old_outputs.pop(0)

        denoised = get_denoised(model_output_prime, self._ts)
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
        self.old_outputs: List[Tensor] = []


__all__ = [
    "PLMSSampler",
]
