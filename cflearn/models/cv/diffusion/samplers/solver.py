import math
import torch

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Optional

from .protocol import ISampler
from .protocol import IDiffusion


@ISampler.register("solver")
class DPMSolver(ISampler):
    def __init__(
        self,
        model: IDiffusion,
        *,
        schedule: str = "linear",
        t0: float = 1.0e-4,
        tT: Optional[float] = None,
        order: int = 3,
        skip_type: str = "logSNR",
        fast_version: bool = True,
        default_steps: int = 20,
    ):
        if model.parameterization != "eps":
            raise ValueError("only `eps` parameterization is supported in `ddim`")
        super().__init__(model)
        if schedule not in ["linear", "cosine"]:
            raise ValueError("only `linear` & `cosine` can be used as `schedule`")
        self.beta_0 = 0.1
        self.beta_1 = 20
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
        self.default_tT = 0.9946 if schedule == "cosine" else 1.0
        self.schedule = schedule
        self.t0 = t0
        self.tT = tT
        self.order = order
        self.skip_type = skip_type
        self.fast_version = fast_version
        self.default_steps = default_steps

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(
            t0=self.t0,
            tT=self.tT,
            order=self.order,
            skip_type=self.skip_type,
            fast_version=self.fast_version,
        )

    def sample_step(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        step: int,
        total_step: int,
        *,
        t0: float = 1.0e-4,
        tT: Optional[float] = None,
        order: int = 3,
        skip_type: str = "logSNR",
        fast_version: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        if tT is None:
            tT = self.default_tT
        if step == 0:
            self._reset_buffers(total_step, t0, tT, order, skip_type, fast_version)
        if step >= len(self.orders):
            return image
        b = image.shape[0]
        device = image.device
        order = self.orders[step]
        vec_s = torch.ones(b, device=device) * self.timesteps[step]
        vec_t = torch.ones(b, device=device) * self.timesteps[step + 1]
        if order == 1:
            return self._order1_update(image, cond, vec_s, vec_t)
        if order == 2:
            return self._order2_update(image, cond, vec_s, vec_t)
        if order == 3:
            return self._order3_update(image, cond, vec_s, vec_t)
        raise ValueError(f"unrecognized order '{order}' occurred")

    # internal

    def _denoise(self, x: Tensor, t: Tensor, cond: Optional[Tensor]) -> Tensor:
        t = self.model.t * torch.max(t - 1.0 / self.model.t, torch.zeros_like(t))
        return self.model.denoise(x, t, cond)

    def _order1_update(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        vec_s: Tensor,
        vec_t: Tensor,
    ) -> Tensor:
        dims = len(image.shape) - 1
        lambda_s = self._marginal_lambda(vec_s)
        lambda_t = self._marginal_lambda(vec_t)
        h = lambda_t - lambda_s
        log_alpha_s = self._marginal_log_mean_coef(vec_s)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_t = self._marginal_std(vec_t)
        phi_1 = torch.expm1(h)
        noise_s = self._denoise(image, vec_s, cond)
        return (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
        )

    def _order2_update(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        vec_s: Tensor,
        vec_t: Tensor,
    ) -> Tensor:
        r1 = 0.5
        dims = len(image.shape) - 1
        lambda_s = self._marginal_lambda(vec_s)
        lambda_t = self._marginal_lambda(vec_t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = self._inverse_lambda(lambda_s1)
        log_alpha_s = self._marginal_log_mean_coef(vec_s)
        log_alpha_s1 = self._marginal_log_mean_coef(s1)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_s1 = self._marginal_std(s1)
        sigma_t = self._marginal_std(vec_t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        noise_s = self._denoise(image, vec_s, cond)
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
        )
        noise_s1 = self._denoise(x_s1, s1, cond)
        return (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
            - (0.5 / r1)
            * (sigma_t * phi_1)[(...,) + (None,) * dims]
            * (noise_s1 - noise_s)
        )

    def _order3_update(
        self,
        image: Tensor,
        cond: Optional[Tensor],
        vec_s: Tensor,
        vec_t: Tensor,
    ) -> Tensor:
        r1 = 1.0 / 3.0
        r2 = 2.0 / 3.0
        dims = len(image.shape) - 1
        lambda_s = self._marginal_lambda(vec_s)
        lambda_t = self._marginal_lambda(vec_t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = self._inverse_lambda(lambda_s1)
        s2 = self._inverse_lambda(lambda_s2)
        log_alpha_s = self._marginal_log_mean_coef(vec_s)
        log_alpha_s1 = self._marginal_log_mean_coef(s1)
        log_alpha_s2 = self._marginal_log_mean_coef(s2)
        log_alpha_t = self._marginal_log_mean_coef(vec_t)
        sigma_s1 = self._marginal_std(s1)
        sigma_s2 = self._marginal_std(s2)
        sigma_t = self._marginal_std(vec_t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
        phi_2 = torch.expm1(h) / h - 1.0

        noise_s = self._denoise(image, vec_s, cond)
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
        )
        noise_s1 = self._denoise(x_s1, s1, cond)
        x_s2 = (
            torch.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_s2 * phi_12)[(...,) + (None,) * dims] * noise_s
            - r2
            / r1
            * (sigma_s2 * phi_22)[(...,) + (None,) * dims]
            * (noise_s1 - noise_s)
        )
        noise_s2 = self._denoise(x_s2, s2, cond)
        return (
            torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * image
            - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
            - (1.0 / r2)
            * (sigma_t * phi_2)[(...,) + (None,) * dims]
            * (noise_s2 - noise_s)
        )

    def _reset_buffers(
        self,
        total_step: int,
        t0: float,
        tT: float,
        order: int,
        skip_type: str,
        fast_version: bool,
    ) -> None:
        if fast_version:
            K = total_step // 3 + 1
            if total_step % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif total_step % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
            timesteps = self._get_time_steps("logSNR", t0, tT, K)
            self.orders = orders
            self.timesteps = timesteps
        else:
            N_steps = total_step // order
            self.orders = [order] * N_steps
            self.timesteps = self._get_time_steps(skip_type, t0, tT, N_steps)

    def _get_time_steps(self, skip_type: str, t0: float, tT: float, N: int) -> Tensor:
        device = self.model.device
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
        if self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if self.schedule == "cosine":
            log_alpha_t = torch.log(
                torch.cos((t + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0)
            )
            log_alpha_t = log_alpha_t - self.cosine_log_alpha_0
            return log_alpha_t
        raise ValueError(f"unrecognized schedule '{self.schedule}' occurred")

    def _marginal_std(self, t: Tensor) -> Tensor:
        return torch.sqrt(1.0 - torch.exp(2.0 * self._marginal_log_mean_coef(t)))

    def _marginal_lambda(self, t: Tensor) -> Tensor:
        log_mean_coef = self._marginal_log_mean_coef(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coef))
        return log_mean_coef - log_std

    def _inverse_lambda(self, lamb: Tensor) -> Tensor:
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
