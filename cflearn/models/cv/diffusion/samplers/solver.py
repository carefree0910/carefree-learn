import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .ddim import DDIMQSampler
from .utils import append_dims
from .utils import interpolate_fn
from .schema import ISampler
from .schema import IQSampler
from .schema import IDiffusion
from .schema import UncondSamplerMixin
from ..utils import cond_type
from .....misc.toolkit import get_device


@ISampler.register("solver")
class DPMSolver(ISampler, UncondSamplerMixin):
    def __init__(
        self,
        model: IDiffusion,
        *,
        schedule: str = "discrete",
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        t0: Optional[float] = None,
        tT: Optional[float] = None,
        order: int = 2,
        skip_type: str = "time_uniform",
        default_steps: int = 25,
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
        predict_x0: bool = True,
        thresholding: bool = False,
        threshold_max_val: float = 1.0,
    ):
        if model.parameterization != "eps":
            raise ValueError("only `eps` parameterization is supported in `DPMSolver`")
        super().__init__(model)
        if schedule not in ["discrete", "linear", "cosine"]:
            msg = "only (`discrete` | `linear` | `cosine`) can be used as `schedule`"
            raise ValueError(msg)
        # schedule
        if schedule == "discrete":
            default_tT = 1.0
            log_alphas = 0.5 * torch.log(model.alphas_cumprod)
            self.total_N = len(log_alphas)
            self.t_array = torch.linspace(0.0, 1.0, len(log_alphas) + 1)[1:].view(1, -1)
            self.log_alpha_array = log_alphas.view(1, -1)
        else:
            default_tT = 0.9946 if schedule == "cosine" else 1.0
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(
                math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
        # properties
        self.schedule = schedule
        self.unconditional_cond = unconditional_cond
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.t0 = 1.0 / self.total_N if t0 is None else t0
        self.tT = default_tT if tT is None else tT
        self.order = order
        self.skip_type = skip_type
        self.default_steps = default_steps
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.threshold_max_val = threshold_max_val

    @property
    def q_sampler(self) -> IQSampler:
        return self._q_sampler

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(
            unconditional_cond=self.unconditional_cond,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
            t0=self.t0,
            tT=self.tT,
            order=self.order,
            skip_type=self.skip_type,
        )

    def sample_step(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        *,
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        t0: float = 1.0e-4,
        tT: float = 1.0,
        order: int = 2,
        skip_type: str = "time_uniform",
        **kwargs: Any,
    ) -> Tensor:
        if step == 0:
            self._reset_buffers(
                total_step,
                unconditional_cond,
                unconditional_guidance_scale,
                t0,
                tT,
                skip_type,
            )
        b = image.shape[0]
        vec_t = self.timesteps[step].to(image).expand(b)
        if not self.t_prev_list:
            self.t_prev_list.append(vec_t)
            self.model_prev_list.append(
                self._model_fn(image, vec_t, cond, step, total_step)
            )
            return image
        if len(self.t_prev_list) < order:
            image = self._multistep_update(image, vec_t, len(self.t_prev_list))
            self.t_prev_list.append(vec_t)
            self.model_prev_list.append(
                self._model_fn(image, vec_t, cond, step, total_step)
            )
            return image
        image = self._multistep_update(image, vec_t, order)
        for i in range(order - 1):
            self.t_prev_list[i] = self.t_prev_list[i + 1]
            self.model_prev_list[i] = self.model_prev_list[i + 1]
        self.t_prev_list[-1] = vec_t
        if step < total_step - 1:
            self.model_prev_list[-1] = self._model_fn(
                image, vec_t, cond, step, total_step
            )
        # clear cache at last step
        else:
            self.t_prev_list.clear()
            self.model_prev_list.clear()
        return image

    # internal

    def _model_fn(
        self,
        x: Tensor,
        ts: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
    ) -> Tensor:
        if self.predict_x0:
            return self._image_fn(x, ts, cond, step, total_step)
        return self._noise_fn(x, ts, cond, step, total_step)

    def _noise_fn(
        self,
        x: Tensor,
        ts: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
    ) -> Tensor:
        ts = self.model.t * torch.max(ts - 1.0 / self.model.t, torch.zeros_like(ts))
        return self._uncond_denoise(x, ts, cond, step, total_step)

    def _image_fn(
        self,
        x: Tensor,
        ts: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
    ) -> Tensor:
        eps = self._noise_fn(x, ts, cond, step, total_step)
        ndim = x.dim()
        alpha_t = self._marginal_alpha(ts)
        sigma_t = self._marginal_std(ts)
        x0 = (x - append_dims(sigma_t, ndim) * eps) / append_dims(alpha_t, ndim)
        if self.thresholding:
            p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = append_dims(
                torch.maximum(s, self.threshold_max_val * torch.ones_like(s)),
                ndim,
            )
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def _multistep_update(
        self,
        x: Tensor,
        vec_t: Tensor,
        order: int,
    ) -> Tensor:
        if order == 1:
            return self._first_update(
                x,
                self.t_prev_list[-1],
                vec_t,
                self.model_prev_list[-1],
            )
        if order == 2:
            return self._multistep_second_update(x, vec_t)
        if order == 3:
            return self._multistep_third_update(x, vec_t)
        raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    def _first_update(
        self,
        x: Tensor,
        vec_s: Tensor,
        vec_t: Tensor,
        model_s: Tensor,
    ) -> Tensor:
        ndim = x.dim()
        lambda_s = self._marginal_lambda(vec_s)
        lambda_t = self._marginal_lambda(vec_t)
        h = lambda_t - lambda_s
        log_alpha_s = self._marginal_log_mean_coef(vec_s)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_s = self._marginal_std(vec_s)
        sigma_t = self._marginal_std(vec_t)
        if self.predict_x0:
            phi_1 = torch.expm1(-h)
            alpha_t = torch.exp(log_alpha_t)
            x_t = (
                append_dims(sigma_t / sigma_s, ndim) * x
                - append_dims(alpha_t * phi_1, ndim) * model_s
            )
            return x_t
        phi_1 = torch.expm1(h)
        x_t = (
            append_dims(torch.exp(log_alpha_t - log_alpha_s), ndim) * x
            - append_dims(sigma_t * phi_1, ndim) * model_s
        )
        return x_t

    def _multistep_second_update(self, x: Tensor, vec_t: Tensor) -> Tensor:
        ndim = x.dim()
        t_prev_1, t_prev_0 = self.t_prev_list
        model_prev_1, model_prev_0 = self.model_prev_list
        lambda_prev_1 = self._marginal_lambda(t_prev_1)
        lambda_prev_0 = self._marginal_lambda(t_prev_0)
        lambda_t = self._marginal_lambda(vec_t)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_t = self._marginal_std(vec_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        d1_0 = append_dims(1.0 / r0, ndim) * (model_prev_0 - model_prev_1)

        if self.predict_x0:
            exp_mh_m1 = torch.exp(-h) - 1.0
            sigma_prev_0 = self._marginal_std(t_prev_0)
            alpha_t = torch.exp(log_alpha_t)
            x_t = (
                append_dims(sigma_t / sigma_prev_0, ndim) * x
                - append_dims(alpha_t * exp_mh_m1, ndim) * model_prev_0
                - 0.5 * append_dims(alpha_t * exp_mh_m1, ndim) * d1_0
            )
            return x_t
        exp_h_m1 = torch.exp(h) - 1.0
        log_alpha_prev_0 = self._marginal_log_mean_coef(t_prev_0)
        x_t = (
            append_dims(torch.exp(log_alpha_t - log_alpha_prev_0), ndim) * x
            - append_dims(sigma_t * exp_h_m1, ndim) * model_prev_0
            - 0.5 * append_dims(sigma_t * exp_h_m1, ndim) * d1_0
        )
        return x_t

    def _multistep_third_update(self, x: Tensor, vec_t: Tensor) -> Tensor:
        ndim = x.dim()
        t_prev_2, t_prev_1, t_prev_0 = self.t_prev_list
        model_prev_2, model_prev_1, model_prev_0 = self.model_prev_list
        lambda_prev_2 = self._marginal_lambda(t_prev_2)
        lambda_prev_1 = self._marginal_lambda(t_prev_1)
        lambda_prev_0 = self._marginal_lambda(t_prev_0)
        lambda_t = self._marginal_lambda(vec_t)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_t = self._marginal_std(vec_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        d1_0 = append_dims(1.0 / r0, ndim) * (model_prev_0 - model_prev_1)
        d1_1 = append_dims(1.0 / r1, ndim) * (model_prev_1 - model_prev_2)
        d1 = d1_0 + append_dims(r0 / (r0 + r1), ndim) * (d1_0 - d1_1)
        d2 = append_dims(1.0 / (r0 + r1), ndim) * (d1_0 - d1_1)

        if self.predict_x0:
            exp_mh_m1 = torch.exp(-h) - 1.0
            sigma_prev_0 = self._marginal_std(t_prev_0)
            alpha_t = torch.exp(log_alpha_t)
            x_t = (
                append_dims(sigma_t / sigma_prev_0, ndim) * x
                - append_dims(alpha_t * exp_mh_m1, ndim) * model_prev_0
                + append_dims(alpha_t * (exp_mh_m1 / h + 1.0), ndim) * d1
                - append_dims(alpha_t * ((exp_mh_m1 + h) / h**2 - 0.5), ndim) * d2
            )
            return x_t
        exp_h_m1 = torch.exp(h) - 1.0
        log_alpha_prev_0 = self._marginal_log_mean_coef(t_prev_0)
        x_t = (
            append_dims(torch.exp(log_alpha_t - log_alpha_prev_0), ndim) * x
            - append_dims(sigma_t * exp_h_m1, ndim) * model_prev_0
            - append_dims(sigma_t * (exp_h_m1 / h - 1.0), ndim) * d1
            - append_dims(sigma_t * ((exp_h_m1 - h) / h**2 - 0.5), ndim) * d2
        )
        return x_t

    def _reset_buffers(
        self,
        total_step: int,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        t0: float,
        tT: float,
        skip_type: str,
    ) -> None:
        self._q_sampler = DDIMQSampler(self.model)
        self._q_sampler.reset_buffers("uniform", total_step)
        self.timesteps = self._get_time_steps(skip_type, t0, tT, total_step - 1)
        assert self.timesteps.shape[0] == total_step
        self.t_prev_list: List[Tensor] = []
        self.model_prev_list: List[Tensor] = []
        # unconditional conditioning
        self._reset_uncond_buffers(unconditional_cond, unconditional_guidance_scale)

    def _get_time_steps(self, skip_type: str, t0: float, tT: float, N: int) -> Tensor:
        device = get_device(self.model)
        if skip_type == "logSNR":
            lambda_T = self._marginal_lambda(torch.tensor(tT, device=device))
            lambda_0 = self._marginal_lambda(torch.tensor(t0, device=device))
            logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1, device=device)
            return self._inverse_lambda(logSNR_steps)
        if skip_type == "time_uniform":
            return torch.linspace(tT, t0, N + 1, device=device)
        if skip_type == "time_quadratic":
            t = torch.linspace(t0, tT, 10000000, device=device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1)
            return torch.flip(
                torch.cat(
                    [
                        t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]],
                        tT * torch.ones((1,)).to(device),
                    ],
                    dim=0,
                ),
                dims=[0],
            )
        raise ValueError(f"unrecognized skip_type '{skip_type}' occurred")

    def _marginal_log_mean_coef(self, t: Tensor) -> Tensor:
        if self.schedule == "discrete":
            t = t.view(-1, 1)
            self.t_array = self.t_array.to(t)
            self.log_alpha_array = self.log_alpha_array.to(t)
            return interpolate_fn(t, self.t_array, self.log_alpha_array).view(-1)
        if self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if self.schedule == "cosine":
            log_alpha_t = torch.log(
                torch.cos((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            log_alpha_t = log_alpha_t - self.cosine_log_alpha_0
            return log_alpha_t
        raise ValueError(f"unrecognized schedule '{self.schedule}' occurred")

    def _marginal_alpha(self, t: Tensor) -> Tensor:
        return torch.exp(self._marginal_log_mean_coef(t))

    def _marginal_std(self, t: Tensor) -> Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self._marginal_log_mean_coef(t)))

    def _marginal_lambda(self, t: Tensor) -> Tensor:
        log_mean_coef = self._marginal_log_mean_coef(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coef))
        return log_mean_coef - log_std

    def _inverse_lambda(self, lamb: Tensor) -> Tensor:
        if self.schedule == "discrete":
            self.t_array = self.t_array.to(lamb)
            self.log_alpha_array = self.log_alpha_array.to(lamb)
            zeros = torch.zeros((1,), dtype=lamb.dtype, device=lamb.device)
            log_alpha = (-0.5 * torch.logaddexp(zeros, -2.0 * lamb)).view(-1, 1)
            t = interpolate_fn(
                log_alpha,
                torch.flip(self.log_alpha_array, [1]),
                torch.flip(self.t_array, [1]),
            )
            return t.view(-1)
        if self.schedule == "linear":
            tmp = (
                2.0
                * (self.beta_1 - self.beta_0)
                * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            )
            delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        if self.schedule == "cosine":
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t = (
                torch.arccos(torch.exp(log_alpha + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            return t
        raise ValueError(f"unrecognized schedule '{self.schedule}' occurred")


__all__ = [
    "DPMSolver",
]
