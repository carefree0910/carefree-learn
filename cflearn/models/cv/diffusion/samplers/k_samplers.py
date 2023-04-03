import torch

import numpy as np

from abc import abstractmethod
from abc import ABCMeta

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from typing import Protocol

from .utils import append_dims
from .utils import append_zero
from .schema import ISampler
from .schema import IQSampler
from .schema import IDiffusion
from .schema import UncondSamplerMixin
from ..utils import cond_type
from ..utils import extract_to

try:
    from scipy import integrate
except:
    integrate = None


def to_d(image: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (image - denoised) / append_dims(sigma, image.ndim)


def get_ancestral_step(
    sigma_from: float,
    sigma_to: float,
    eta: float = 1.0,
) -> Tuple[float, float]:
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


class IGetDenoised(Protocol):
    def __call__(self, img: Tensor, sigma: Tensor) -> Tensor:
        pass


class KQSampler(IQSampler):
    def q_sample(
        self,
        net: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        w_noise = extract_to(self.sigmas, timesteps, net.ndim)
        if noise is None:
            noise = torch.randn_like(net)
        net = net + w_noise * noise
        return net

    def reset_buffers(self, sigmas: Tensor) -> None:  # type: ignore
        self.sigmas = sigmas


class KSamplerMixin(ISampler, UncondSamplerMixin, metaclass=ABCMeta):
    def __init__(
        self,
        model: IDiffusion,
        *,
        default_quantize: bool = False,
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        default_steps: int = 25,
    ):
        if model.parameterization not in ("eps", "v"):
            msg = "only `v` / `eps` parameterization is supported in `k_samplers`"
            raise ValueError(msg)
        super().__init__(model)
        self.default_quantize = default_quantize
        self.unconditional_cond = unconditional_cond
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.default_steps = default_steps

    # abstract

    @abstractmethod
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_denoised: IGetDenoised,
        *,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        **kwargs: Any,
    ) -> Tensor:
        pass

    # inheritance

    @property
    def q_sampler(self) -> KQSampler:
        return self._q_sampler

    @property
    def sample_kwargs(self) -> Dict[str, Any]:
        return dict(
            quantize=self.default_quantize,
            unconditional_cond=self.unconditional_cond,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
        )

    def sample_step(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        *,
        quantize: bool = False,
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> Tensor:
        if step == 0 and not self.initialized:
            self._reset_buffers(
                total_step,
                quantize,
                unconditional_cond,
                unconditional_guidance_scale,
            )
            image = image * self.sigmas[0]
        if step == total_step - 1:
            return image

        def get_denoised(img: Tensor, sigma: Tensor) -> Tensor:
            ndim = img.ndim
            sigma = sigma * s_in
            ts = self._sigma_to_t(sigma, quantize)
            if self.model.parameterization == "eps":
                c_in = append_dims(
                    1.0 / (sigma**2 + self.sigma_data**2) ** 0.5,
                    ndim,
                )
                c_out = append_dims(sigma, ndim)
                eps = self._uncond_denoise(img * c_in, ts, cond, step, total_step)
                return img - eps * c_out
            if self.model.parameterization == "v":
                c_in = append_dims(
                    1.0 / (sigma**2 + self.sigma_data**2) ** 0.5,
                    ndim,
                )
                c_out = append_dims(
                    sigma
                    * self.sigma_data
                    / (sigma**2 + self.sigma_data**2) ** 0.5,
                    ndim,
                )
                c_skip = append_dims(
                    self.sigma_data**2 / (sigma**2 + self.sigma_data**2),
                    ndim,
                )
                v = self._uncond_denoise(img * c_in, ts, cond, step, total_step)
                return img * c_skip - v * c_out
            raise ValueError(
                f"unrecognized parameterization `{self.model.parameterization}` occurred"
            )

        s_in = image.new_ones([image.shape[0]])
        return self.sample_step_core(
            image,
            cond,
            step,
            total_step,
            get_denoised,
            quantize=quantize,
            unconditional_cond=unconditional_cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            **kwargs,
        )

    # internal

    def _t_to_sigma(self, ts: Tensor) -> Tensor:
        log_base = self.log_sigmas_base
        low_idx, high_idx, w = ts.floor().long(), ts.ceil().long(), ts.frac()
        log_sigma = (1.0 - w) * log_base[low_idx] + w * log_base[high_idx]
        return log_sigma.exp()

    def _sigma_to_t(self, sigmas: Tensor, quantize: bool) -> Tensor:
        quantize = self.quantize if quantize is None else quantize
        log_sigmas = sigmas.log()
        dists = log_sigmas - self.log_sigmas_base[:, None]
        if quantize:
            return dists.abs().argmin(dim=0).view(sigmas.shape)
        low_idx = (
            dists.ge(0.0)
            .cumsum(dim=0)
            .argmax(dim=0)
            .clamp(max=self.log_sigmas_base.shape[0] - 2)
        )
        high_idx = low_idx + 1
        low, high = self.log_sigmas_base[low_idx], self.log_sigmas_base[high_idx]
        w = (low - log_sigmas) / (low - high)
        w = w.clamp(0.0, 1.0)
        t = (1.0 - w) * low_idx + w * high_idx
        return t.view(sigmas.shape)

    def _reset_buffers(
        self,
        total_step: int,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        alphas = self.model.alphas_cumprod
        self.sigmas_base = ((1.0 - alphas) / alphas) ** 0.5
        self.log_sigmas_base = self.sigmas_base.log()
        t_max = len(self.sigmas_base) - 1
        ts = torch.linspace(t_max, 0, total_step, device=self.sigmas_base.device)
        self.sigmas = append_zero(self._t_to_sigma(ts)).to(alphas.dtype)
        self.sigma_data = 1.0
        self.quantize = quantize
        # q sampling
        self._q_sampler = KQSampler(self.model)
        self._q_sampler.reset_buffers(self.sigmas.flip(0))
        # unconditional conditioning
        self._reset_uncond_buffers(unconditional_cond, unconditional_guidance_scale)
        # set flag
        self.initialized = True


def klms_coef(order: int, t: np.ndarray, i: int, j: int) -> float:
    def fn(tau: float) -> float:
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    if integrate is None:
        raise ValueError("`scipy` is needed for `KLMSSampler`")
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1.0e-4)[0]


@ISampler.register("klms")
class KLMSSampler(KSamplerMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_denoised: IGetDenoised,
        *,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        **kwargs: Any,
    ) -> Tensor:
        order = 4
        denoised = get_denoised(image, self.sigmas[step])
        d = to_d(image, self.sigmas[step], denoised)
        self.ds.append(d)
        if len(self.ds) > order:
            self.ds.pop(0)
        current_order = min(step + 1, order)
        for i, d in enumerate(reversed(self.ds)):
            coef = klms_coef(current_order, self.sigmas_numpy, step, i)
            image = image + coef * d
        return image

    def _reset_buffers(
        self,
        total_step: int,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        super()._reset_buffers(
            total_step,
            quantize,
            unconditional_cond,
            unconditional_guidance_scale,
        )
        self.ds: List[Tensor] = []
        self.sigmas_numpy = self.sigmas.detach().cpu().numpy()


@ISampler.register("k_euler")
class KEulerSampler(KSamplerMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_denoised: IGetDenoised,
        *,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        **kwargs: Any,
    ) -> Tensor:
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = float("inf")
        s_noise = 1.0
        sigma = self.sigmas[step]
        if not s_tmin <= sigma.item() <= s_tmax:
            gamma = 0.0
        else:
            gamma = min(s_churn / (total_step - 1), 2**0.5 - 1)
        eps = torch.randn_like(image) * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            image = image + eps * (sigma_hat**2 - sigma**2) ** 0.5
        denoised = get_denoised(image, sigma_hat)
        d = to_d(image, sigma_hat, denoised)
        dt = self.sigmas[step + 1] - sigma_hat
        image = image + d * dt
        return image


@ISampler.register("k_euler_a")
class KEulerAncestralSampler(KSamplerMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_denoised: IGetDenoised,
        *,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        **kwargs: Any,
    ) -> Tensor:
        eta = 1.0
        s_noise = 1.0
        sigma = self.sigmas[step]
        next_sigma = self.sigmas[step + 1]
        denoised = get_denoised(image, sigma)
        sigma_down, sigma_up = get_ancestral_step(
            sigma.item(),
            next_sigma.item(),
            eta=eta,
        )
        d = to_d(image, sigma, denoised)
        dt = sigma_down - sigma
        image = image + d * dt
        if next_sigma.item() > 0:
            image = image + torch.randn_like(image) * s_noise * sigma_up
        return image


@ISampler.register("k_heun")
class KHeunSampler(KSamplerMixin):
    def sample_step_core(
        self,
        image: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
        get_denoised: IGetDenoised,
        *,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
        **kwargs: Any,
    ) -> Tensor:
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = float("inf")
        s_noise = 1.0
        sigma = self.sigmas[step]
        next_sigma = self.sigmas[step + 1]
        if not s_tmin <= sigma.item() <= s_tmax:
            gamma = 0.0
        else:
            gamma = min(s_churn / (total_step - 1), 2**0.5 - 1)
        eps = torch.randn_like(image) * s_noise
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            image = image + eps * (sigma_hat**2 - sigma**2) ** 0.5
        denoised = get_denoised(image, sigma_hat)
        d = to_d(image, sigma_hat, denoised)
        dt = next_sigma - sigma_hat
        if next_sigma.item() == 0:
            image = image + d * dt
        else:
            x_2 = image + d * dt
            denoised_2 = get_denoised(x_2, next_sigma)
            d_2 = to_d(x_2, next_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            image = image + d_prime * dt
        return image


__all__ = [
    "KLMSSampler",
    "KEulerSampler",
    "KEulerAncestralSampler",
    "KHeunSampler",
]
