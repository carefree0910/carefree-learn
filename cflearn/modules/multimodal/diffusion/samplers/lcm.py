from torch import Tensor
from typing import Any
from typing import Tuple
from typing import Optional

from .ddim import DDIMSampler
from .ddim import IGetModelOutput
from .ddim import IGetDenoised
from .schema import ISampler
from ..utils import cond_type
from ..utils import get_timesteps


@ISampler.register("lcm")
class LCMSampler(DDIMSampler):
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
        denoised = super().sample_step_core(
            image,
            cond,
            step,
            total_step,
            get_model_output,
            get_denoised,
            eta=eta,
            discretize=discretize,
            unconditional_cond=unconditional_cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
            **kwargs,
        )
        if step < total_step - 1:
            denoised = self.q_sample(
                denoised,
                get_timesteps(self._t_index_prev, denoised.shape[0], denoised.device),
            )
        return denoised

    def _get_denoised_and_pred_x0(
        self,
        model_output: Tensor,
        ts: Tensor,
        image: Tensor,
        quantize_denoised: bool,
        temperature: float,
        noise_dropout: float,
    ) -> Tuple[Tensor, Tensor]:
        # c_skip, c_out
        sigma_data = 0.5
        t_div = self._t / 0.1
        c_skip = sigma_data**2 / (t_div**2 + sigma_data**2)
        c_out = t_div / (t_div**2 + sigma_data**2) ** 0.5
        # denoise
        _, pred_x0 = self._get_pred_x0(model_output, ts, image, quantize_denoised)
        denoised = c_out * pred_x0 + c_skip * image
        return denoised, pred_x0


__all__ = [
    "LCMSampler",
]
