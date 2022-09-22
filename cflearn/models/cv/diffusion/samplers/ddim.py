import torch

import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Protocol

from .protocol import ISampler
from .protocol import IDiffusion
from .protocol import QSampleMixin
from .protocol import UncondSamplerMixin
from ..utils import get_timesteps
from ...ae.vq import AutoEncoderVQModel


class IGetEPSPred(Protocol):
    def __call__(self, image: Tensor, ts: Tensor) -> Tensor:
        pass


class IGetDenoised(Protocol):
    def __call__(self, eps: Tensor) -> Tensor:
        pass


class DDIMMixin(QSampleMixin, ISampler, UncondSamplerMixin, metaclass=ABCMeta):
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

    # abstract

    @abstractmethod
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
        pass

    # inheritance

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
        return self.sample_step_core(
            image,
            cond,
            step,
            total_step,
            lambda img, ts: self._uncond_denoise(img, ts, cond),
            lambda eps: self._get_denoised_and_pred_x0(
                eps,
                image,
                quantize_denoised,
                temperature,
                noise_dropout,
            )[0],
            eta=eta,
            discretize=discretize,
            unconditional_cond=unconditional_cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            temperature=temperature,
            noise_dropout=noise_dropout,
            quantize_denoised=quantize_denoised,
            **kwargs,
        )

    # internal

    def _register_temp_buffers(self, image: Tensor, step: int, total_step: int) -> None:
        b = image.shape[0]
        device = image.device
        index = total_step - step - 1
        t = self.ddim_timesteps[index]
        t_next = self.ddim_timesteps[max(index - 1, 0)]
        self._ts = get_timesteps(t, b, device)
        self._ts_next = get_timesteps(t_next, b, device)
        extract = lambda base: torch.full(
            (b, 1, 1, 1),
            base[index],
            dtype=image.dtype,
            device=device,
        )
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
        self._reset_q_buffers(discretize, total_step)
        alphas = self.model.alphas_cumprod
        self.ddim_alphas = self.q_alphas
        self.ddim_alphas_prev = torch.tensor(
            [alphas[0]] + alphas[self.q_timesteps[:-1]].tolist(),
            dtype=alphas.dtype,
            device=self.model.device,
        )
        self.ddim_sigmas = eta * torch.sqrt(
            (1.0 - self.ddim_alphas_prev)
            / (1.0 - self.ddim_alphas)
            * (1.0 - self.ddim_alphas / self.ddim_alphas_prev)
        )
        self.ddim_sqrt_one_minus_alphas = self.q_sqrt_one_minus_alphas
        self.ddim_timesteps = self.q_timesteps.tolist()
        # unconditional conditioning
        self._reset_uncond_buffers(unconditional_cond, unconditional_guidance_scale)


@ISampler.register("ddim")
class DDIMSampler(DDIMMixin):
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
        denoised = get_denoised(eps_pred)
        return denoised


__all__ = [
    "DDIMSampler",
]
