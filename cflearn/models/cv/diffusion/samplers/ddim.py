import math
import torch

import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict

from .protocol import ISampler
from .protocol import IDiffusion
from ..utils import get_timesteps
from ...ae.vq import AutoEncoderVQModel
from .....protocol import tensor_dict_type


@ISampler.register("ddim")
class DDIMSampler(ISampler):
    def __init__(
        self,
        model: IDiffusion,
        *,
        eta: float = 0.0,
        discretize: str = "uniform",
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
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.quantize_denoised = quantize_denoised
        self.default_steps = default_steps

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(
            eta=self.eta,
            discretize=self.discretize,
            temperature=self.temperature,
            noise_dropout=self.noise_dropout,
            quantize_denoised=self.quantize_denoised,
        )

    def sample_step(
        self,
        image: Tensor,
        cond_kw: tensor_dict_type,
        step: int,
        total_step: int,
        *,
        eta: float = 0.0,
        discretize: str = "uniform",
        temperature: float = 1.0,
        noise_dropout: float = 1.0,
        quantize_denoised: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if step == 0:
            self._reset_buffers(eta, discretize, total_step)
        b = image.shape[0]
        device = image.device
        index = total_step - step - 1
        t = self.ddim_timesteps[index]
        ts = get_timesteps(t, b, device)
        extract = lambda base: torch.full((b, 1, 1, 1), base[index], device=device)
        alphas_t = extract(self.ddim_alphas)
        alphas_prev_t = extract(self.ddim_alphas_prev)
        sigmas_t = extract(self.ddim_sigmas)
        sqrt_one_minus_alphas_t = extract(self.ddim_sqrt_one_minus_alphas)
        # execute
        noise_pred = self.model.denoise(image, ts, cond_kw)
        pred_x0 = (image - sqrt_one_minus_alphas_t * noise_pred) / alphas_t.sqrt()
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
        direction = (1.0 - alphas_prev_t - sigmas_t**2).sqrt() * noise_pred
        noise = sigmas_t * torch.randn_like(image) * temperature
        if noise_dropout > 0:
            noise = F.dropout(noise, p=noise_dropout)
        net = alphas_prev_t.sqrt() * pred_x0 + direction + noise
        return net

    def _reset_buffers(self, eta: float, discretize: str, total_step: int) -> None:
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


__all__ = [
    "DDIMSampler",
]
