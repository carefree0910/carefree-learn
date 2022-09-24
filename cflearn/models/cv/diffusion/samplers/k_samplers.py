import torch

import numpy as np

from abc import abstractmethod
from abc import ABCMeta

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol

from .utils import append_dims
from .utils import append_zero
from .protocol import ISampler
from .protocol import IQSampler
from .protocol import IDiffusion
from .protocol import UncondSamplerMixin
from ..utils import extract_to

try:
    from scipy import integrate
except:
    integrate = None


def to_d(image: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (image - denoised) / append_dims(sigma, image.ndim)


class IGetDenoised(Protocol):
    def __call__(self, img: Tensor) -> Tensor:
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
        default_steps: int = 50,
    ):
        if model.parameterization != "eps":
            raise ValueError("only `eps` parameterization is supported in `k_samplers`")
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
        cond: Optional[Tensor],
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
        cond: Optional[Tensor],
        step: int,
        total_step: int,
        *,
        quantize: bool = False,
        unconditional_cond: Optional[Any] = None,
        unconditional_guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> Tensor:
        if step == 0:
            self._reset_buffers(
                total_step,
                quantize,
                unconditional_cond,
                unconditional_guidance_scale,
            )
            image = image * self.sigmas[0]
        if step == total_step - 1:
            return image

        def get_denoised(img: Tensor) -> Tensor:
            sigmas = self.sigmas[step] * image.new_ones([image.shape[0]])
            ts = self._sigma_to_t(sigmas, quantize)
            c_in = append_dims(
                1.0 / (sigmas**2 + self.sigma_data**2) ** 0.5, img.ndim
            )
            c_out = append_dims(sigmas, img.ndim)
            eps = self._uncond_denoise(img * c_in, ts, cond)
            return img - eps * c_out

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
        low_idx, high_idx, w = ts.floor().long(), ts.ceil().long(), ts.frac()
        return (1.0 - w) * self.sigmas_base[low_idx] + w * self.sigmas_base[high_idx]

    def _sigma_to_t(self, sigmas: Tensor, quantize: bool) -> Tensor:
        dists = torch.abs(sigmas - self.sigmas_base[:, None])
        if quantize:
            return torch.argmin(dists, dim=0).view(sigmas.shape)
        top2 = torch.topk(dists, dim=0, k=2, largest=False).indices
        low_idx, high_idx = torch.sort(top2, dim=0)[0]
        low, high = self.sigmas_base[low_idx], self.sigmas_base[high_idx]
        w = (low - sigmas) / (low - high)
        w = w.clamp(0, 1)
        ts = (1.0 - w) * low_idx + w * high_idx
        return ts.view(sigmas.shape)

    def _reset_buffers(
        self,
        total_step: int,
        quantize: bool,
        unconditional_cond: Optional[Any],
        unconditional_guidance_scale: float,
    ) -> None:
        alphas = self.model.alphas_cumprod
        self.sigmas_base = ((1.0 - alphas) / alphas) ** 0.5
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
        cond: Optional[Tensor],
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
        denoised = get_denoised(image)
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


__all__ = [
    "KLMSSampler",
]
