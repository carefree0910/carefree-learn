import math
import torch

import numpy as np
import torch.nn.functional as F

from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .protocol import ISampler
from .protocol import IDiffusion
from ..utils import get_timesteps
from ...ae.vq import AutoEncoderVQModel


class DDIMMixin(ISampler, metaclass=ABCMeta):
    def __init__(
        self,
        model: IDiffusion,
        *,
        eta: float = 0.0,
        discretize: str = "uniform",
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        temperature: float = 1.0,
        noise_dropout: float = 0.0,
        quantize_denoised: bool = False,
        default_steps: int = 50,
    ):
        if model.parameterization != "eps":
            raise ValueError("only `eps` parameterization is supported in `ddim`")
        super().__init__(model)
        self.eta = eta
        self.discretize = discretize
        self.unconditional_cond = unconditional_cond
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.quantize_denoised = quantize_denoised
        self.default_steps = default_steps

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(
            eta=self.eta,
            discretize=self.discretize,
            unconditional_cond=self.unconditional_cond,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
            temperature=self.temperature,
            noise_dropout=self.noise_dropout,
            quantize_denoised=self.quantize_denoised,
        )

    def _denoise(self, image: Tensor, ts: Tensor, cond: Optional[Tensor]) -> Tensor:
        if cond is None or self.uncond is None:
            return self.model.denoise(image, ts, cond)
        uncond = self.uncond.repeat_interleave(cond.shape[0], dim=0)
        cond2 = torch.cat([uncond, cond])
        image2 = torch.cat([image, image])
        ts2 = torch.cat([ts, ts])
        eps_uncond, eps = self.model.denoise(image2, ts2, cond2).chunk(2)
        return eps_uncond + self.uncond_guidance_scale * (eps - eps_uncond)

    def _register_temp_buffers(self, image: Tensor, step: int, total_step: int) -> None:
        b = image.shape[0]
        device = image.device
        index = total_step - step - 1
        t = self.ddim_timesteps[index]
        t_next = self.ddim_timesteps[max(index - 1, 0)]
        self._ts = get_timesteps(t, b, device)
        self._ts_next = get_timesteps(t_next, b, device)
        extract = lambda base: torch.full((b, 1, 1, 1), base[index], device=device)
        self._at = extract(self.ddim_alphas)
        self._a_prev_t = extract(self.ddim_alphas_prev)
        self._sigmas_t = extract(self.ddim_sigmas)
        self._sqrt_one_minus_at = extract(self.ddim_sqrt_one_minus_alphas)

    def _get_denoised_and_pred_x0(
        self,
        eps_pred: Tensor,
        image: Tensor,
        quantize_denoised: bool,
        temperature: float,
        noise_dropout: float,
    ) -> Tuple[Tensor, Tensor]:
        pred_x0 = (image - self._sqrt_one_minus_at * eps_pred) / self._at.sqrt()
        if quantize_denoised:
            err_fmt = "only {} can use `quantize_denoised`"
            first_stage = getattr(self.model, "first_stage", None)
            if first_stage is None:
                raise ValueError(err_fmt.format("model with `first_stage` model"))
            vq = first_stage.core
            if not isinstance(vq, AutoEncoderVQModel):
                raise ValueError(
                    err_fmt.format(
                        "model with `AutoEncoderVQModel` as the `first_stage` model"
                    )
                )
            pred_x0 = vq.codebook(pred_x0).z_q
        direction = (1.0 - self._a_prev_t - self._sigmas_t**2).sqrt() * eps_pred
        noise = self._sigmas_t * torch.randn_like(image) * temperature
        if noise_dropout > 0:
            noise = F.dropout(noise, p=noise_dropout)
        denoised = self._a_prev_t.sqrt() * pred_x0 + direction + noise
        return denoised, pred_x0

    def _reset_buffers(
        self,
        eta: float,
        discretize: str,
        total_step: int,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        # discretize time steps
        if discretize == "uniform":
            span = self.model.t // total_step
            ddim_timesteps = np.array(list(range(0, self.model.t, span)))
        elif discretize == "quad":
            end = math.sqrt(self.model.t * 0.8)
            ddim_timesteps = (np.linspace(0, end, total_step) ** 2).astype(int)
        else:
            raise ValueError(f"unrecognized discretize method '{discretize}' occurred")
        ddim_timesteps += 1
        # calculate parameters
        alphas = self.model.alphas_cumprod
        self.ddim_alphas = alphas[ddim_timesteps]
        self.ddim_alphas_prev = torch.tensor(
            [alphas[0]] + alphas[ddim_timesteps[:-1]].tolist(),
            device=self.model.device,
        )
        self.ddim_sigmas = eta * torch.sqrt(
            (1.0 - self.ddim_alphas_prev)
            / (1.0 - self.ddim_alphas)
            * (1.0 - self.ddim_alphas / self.ddim_alphas_prev)
        )
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)
        self.ddim_timesteps = ddim_timesteps.tolist()
        # unconditional conditioning
        if unconditional_cond is None or self.model.condition_model is None:
            self.uncond = None
            self.uncond_guidance_scale = 0.0
        else:
            self.uncond = self.model._get_cond(unconditional_cond)
            self.uncond_guidance_scale = unconditional_guidance_scale


@ISampler.register("ddim")
class DDIMSampler(DDIMMixin):
    def sample_step(
        self,
        image: Tensor,
        cond: Optional[Tensor],
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
        eps_pred = self._denoise(image, self._ts, cond)
        denoised, _ = self._get_denoised_and_pred_x0(
            eps_pred,
            image,
            quantize_denoised,
            temperature,
            noise_dropout,
        )
        return denoised


__all__ = [
    "DDIMSampler",
]
