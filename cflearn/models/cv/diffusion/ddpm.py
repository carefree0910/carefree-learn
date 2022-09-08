import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.array import to_torch
from cftool.types import tensor_dict_type

from .unet import UNetDiffuser
from .utils import extract_to
from .utils import get_timesteps
from .samplers import ISampler
from .cond_models import condition_models
from ...protocols import GaussianGeneratorMixin
from ....zoo import DLZoo
from ....protocol import ITrainer
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....constants import INPUT_KEY
from ....constants import PREDICTIONS_KEY
from ....misc.toolkit import freeze
from ....misc.internal_ import register_custom_module
from ....misc.internal_ import CustomModule
from ....misc.internal_ import CustomTrainStep
from ....misc.internal_ import CustomTrainStepLoss
from ....modules.blocks import EMA


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    linear_start: float,
    linear_end: float,
    cosine_s: float,
) -> np.ndarray:
    if schedule == "linear":
        betas = (
            np.linspace(
                linear_start**0.5,
                linear_end**0.5,
                timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif schedule == "cosine":
        arange = np.arange(timesteps + 1, dtype=np.float64)
        timesteps = arange / timesteps + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1.0 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0.0, 0.999)
    else:
        lin = np.linspace(linear_start, linear_end, timesteps, dtype=np.float64)
        if schedule == "sqrt_linear":
            betas = lin
        elif schedule == "sqrt":
            betas = lin**0.5
        else:
            raise ValueError(f"unrecognized schedule '{schedule}' occurred")
    return betas


class DDPMStep(CustomTrainStep):
    def loss_fn(
        self,
        m: "DDPM",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
        **kwargs: Any,
    ) -> CustomTrainStepLoss:
        unet_out = forward_results[PREDICTIONS_KEY]
        noise = forward_results[m.noise_key]
        if m.parameterization == "eps":
            target = noise
        elif m.parameterization == "x0":
            target = batch[INPUT_KEY]
        else:
            msg = f"unrecognized parameterization '{m.parameterization}' occurred"
            raise ValueError(msg)

        losses = {}
        if m.loss_type == "l1":
            loss = (unet_out - target).abs()
        elif m.loss_type == "l2":
            loss = F.mse_loss(unet_out, target, reduction="none")
        else:
            raise ValueError(f"unrecognized loss '{m.loss_type}' occurred")
        loss = loss.mean(dim=(1, 2, 3))
        loss_simple = loss
        losses["simple"] = loss_simple.mean().item()

        timesteps = forward_results[m.timesteps_key]
        log_var_t = m.log_var[timesteps].to(unet_out.device)  # type: ignore
        loss_simple = loss_simple / torch.exp(log_var_t) + log_var_t
        if m.learn_log_var:
            losses["gamma"] = loss_simple.mean().item()
            losses["log_var"] = m.log_var.data.mean().item()  # type: ignore

        loss_simple = m.l_simple_weight * loss_simple.mean()
        if m.original_elbo_weight <= 0:
            losses["loss"] = loss_simple.item()
            return CustomTrainStepLoss(loss_simple, losses)

        loss_vlb = (m.lvlb_weights[timesteps] * loss).mean()
        losses["vlb"] = loss_vlb.item()

        loss_vlb = m.original_elbo_weight * loss_vlb
        loss = loss_simple + loss_vlb
        losses["loss"] = loss.item()
        return CustomTrainStepLoss(loss, losses)

    def callback(
        self,
        m: "DDPM",
        trainer: ITrainer,
        batch: tensor_dict_type,
        forward_results: Union[tensor_dict_type, List[tensor_dict_type]],
    ) -> None:
        if m.training and m.unet_ema is not None:
            m.unet_ema()


@register_custom_module("ddpm")
class DDPM(CustomModule, GaussianGeneratorMixin):
    cond_key = "cond"
    noise_key = "noise"
    timesteps_key = "timesteps"
    identity_condition_model = "identity"

    sampler: ISampler

    def __init__(
        self,
        img_size: int,
        # unet
        in_channels: int,
        out_channels: int,
        *,
        start_channels: int = 320,
        num_heads: Optional[int] = 8,
        num_head_channels: Optional[int] = None,
        use_spatial_transformer: bool = True,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        resample_with_conv: bool = True,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        # diffusion
        ema_decay: Optional[float] = None,
        use_num_updates_in_ema: bool = True,
        parameterization: str = "eps",
        ## condition
        condition_type: str = "cross_attn",
        condition_model: Optional[str] = None,
        condition_config: Optional[Dict[str, Any]] = None,
        condition_learnable: bool = False,
        ## noise schedule
        v_posterior: float = 0.0,
        timesteps: int = 1000,
        given_betas: Optional[np.ndarray] = None,
        beta_schedule: str = "linear",
        linear_start: float = 1.0e-4,
        linear_end: float = 2.0e-2,
        cosine_s: float = 8.0e-3,
        ## loss
        loss_type: str = "l2",
        l_simple_weight: float = 1.0,
        original_elbo_weight: float = 0.0,
        learn_log_var: bool = False,
        log_var_init: float = 0.0,
        ## sampling
        sampler: str = "ddim",
        sampler_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        # unet
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = UNetDiffuser(
            in_channels,
            out_channels,
            context_dim=context_dim,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_spatial_transformer=use_spatial_transformer,
            num_transformer_layers=num_transformer_layers,
            signal_dim=signal_dim,
            start_channels=start_channels,
            num_res_blocks=num_res_blocks,
            attention_downsample_rates=attention_downsample_rates,
            dropout=dropout,
            channel_multipliers=channel_multipliers,
            resample_with_conv=resample_with_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            num_classes=num_classes,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
        )
        # ema
        if ema_decay is None:
            self.unet_ema = None
        else:
            self.unet_ema = EMA(
                ema_decay,
                list(self.unet.named_parameters()),
                use_num_updates=use_num_updates_in_ema,
            )
        # condition
        self.condition_type = condition_type
        self.condition_learnable = condition_learnable
        self._initialize_condition_model(
            condition_model,
            condition_config,
            condition_learnable,
        )
        # settings
        self.parameterization = parameterization
        self.v_posterior = v_posterior
        # noise schedule
        self._register_noise_schedule(
            timesteps,
            given_betas,
            beta_schedule,
            linear_start,
            linear_end,
            cosine_s,
        )
        # loss
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.learn_log_var = learn_log_var
        log_var = torch.full(fill_value=log_var_init, size=(self.t,))
        if not learn_log_var:
            self.log_var = log_var
        else:
            self.log_var = nn.Parameter(log_var, requires_grad=True)
        # sampler
        self.switch_sampler(sampler, sampler_config)

    @property
    def can_reconstruct(self) -> bool:
        return True

    @property
    def learnable(self) -> List[nn.Parameter]:
        params = list(self.unet.parameters())
        if self.learn_log_var:
            params.append(self.log_var)
        return params

    @property
    def train_steps(self) -> List[CustomTrainStep]:
        return [DDPMStep("core.learnable")]

    def forward(
        self,
        batch: tensor_dict_type,
        *,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        use_noise: bool = True,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        cond = batch.get(self.cond_key)
        # timesteps
        ts = torch.randint(0, self.t, (net.shape[0],), device=net.device).long()
        if timesteps is None:
            timesteps = ts
        # condition
        if cond is not None and self.condition_model is not None:
            cond = self._get_cond(cond)
        # get input
        net, cond_kw = self._get_input(net, cond)
        # noise
        if noise is None and use_noise:
            noise = torch.randn_like(net)
        if noise is not None:
            net = self._q_sample(net, timesteps, noise)
        # unet
        unet_out = self.unet(net, timesteps=timesteps, **cond_kw)
        return {
            PREDICTIONS_KEY: unet_out,
            self.noise_key: noise,
            self.timesteps_key: timesteps,
        }

    def evaluate_step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: TrainerState,
        weighted_loss_score_fn: Callable[[Dict[str, float]], float],
        trainer: ITrainer,
    ) -> MetricsOutputs:
        train_step = self.train_steps[0]
        # TODO : specify timesteps & noise to make results deterministic
        forward = self.forward(batch)
        losses = train_step.loss_fn(self, trainer, batch, forward).losses
        score = -losses["simple"]
        # no ema
        if self.unet_ema is None:
            return MetricsOutputs(score, losses)
        losses = {f"{k}_ema": v for k, v in losses.items()}
        self.unet_ema.train()
        forward = self.forward(batch)
        losses.update(train_step.loss_fn(self, trainer, batch, forward).losses)
        self.unet_ema.eval()
        return MetricsOutputs(score, losses)

    # api

    def make_sampler(
        self,
        sampler: str,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> ISampler:
        kw = shallow_copy_dict(sampler_config or {})
        kw["model"] = self
        return ISampler.make(sampler, kw)

    def switch_sampler(
        self,
        sampler: str,
        sampler_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sampler = self.make_sampler(sampler, sampler_config)

    def generate_z(self, num_samples: int) -> Tensor:
        shape = num_samples, self.in_channels, self.img_size, self.img_size
        return torch.randn(shape, device=self.device)

    def decode(
        self,
        z: Tensor,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        return self.sampler.sample(
            z,
            cond=cond,
            num_steps=num_steps,
            verbose=verbose,
            **kwargs,
        )

    def sample(
        self,
        num_samples: int,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        sampled = super().sample(
            num_samples,
            cond=cond,
            num_steps=num_steps,
            verbose=verbose,
            **kwargs,
        )
        if clip_output:
            sampled = torch.clip(sampled, -1.0, 1.0)
        return sampled

    def reconstruct(
        self,
        net: Tensor,
        *,
        noise_steps: Optional[int] = None,
        cond: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tensor:
        if noise_steps is None:
            noise_steps = self.t
        ts = get_timesteps(noise_steps - 1, net.shape[0], net.device)
        z = self._q_sample(net, ts, torch.randn_like(net))
        kw = shallow_copy_dict(kwargs)
        kw.update(dict(z=z, cond=cond))
        net = safe_execute(self.decode, kw)
        return net

    def denoise(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond_kw: tensor_dict_type,
    ) -> Tensor:
        return self.unet(image, timesteps=timesteps, **cond_kw)

    # internal

    def _q_sample(self, net: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        num_dim = len(net.shape)
        w_net = extract_to(self.sqrt_alphas_cumprod, timesteps, num_dim)
        w_noise = extract_to(self.sqrt_one_minus_alphas_cumprod, timesteps, num_dim)
        net = w_net * net + w_noise * noise
        return net

    def _get_cond(self, cond: Any) -> Tensor:
        if self.condition_model is None:
            msg = "should not call `get_cond` when `condition_model` is not provided"
            raise ValueError(msg)
        return self.condition_model(cond)

    # return input & condition inputs
    def _get_input(
        self,
        net: Tensor,
        cond: Optional[Tensor],
        *,
        in_decode: bool = False,
    ) -> Tuple[Tensor, tensor_dict_type]:
        if cond is None or self.condition_type is None:
            return net, {}
        msg = f"`cond` should be provided when condition_type='{self.condition_type}'"
        if self.condition_type == "concat":
            if cond is None:
                raise ValueError(msg)
            return torch.cat([net, cond]), {}
        if self.condition_type == "cross_attn":
            return net, {"context": cond}
        if self.condition_type == "adm":
            return net, {"labels": cond}
        raise ValueError(f"unrecognized condition type {self.condition_type} occurred")

    def _make_condition_model(self, key: str, m: nn.Module) -> None:
        condition_base = condition_models.get(key)
        if condition_base is None:
            self.condition_model = m
        else:
            self.condition_model = condition_base(m)

    def _initialize_condition_model(
        self,
        condition_model: Optional[str],
        condition_config: Optional[Dict[str, Any]],
        condition_learnable: bool,
    ) -> None:
        if condition_model is None:
            self.condition_model = None
            return
        if condition_model == self.identity_condition_model:
            self.condition_model = nn.Identity()
            return
        kwargs = condition_config or {}
        kwargs.setdefault("report", False)
        if not condition_learnable:
            kwargs.setdefault("pretrained", True)
        m = DLZoo.load_model(condition_model, **kwargs)
        if not condition_learnable:
            freeze(m)
        self._make_condition_model(condition_model, m)

    def _register_noise_schedule(
        self,
        timesteps: int,
        given_betas: Optional[np.ndarray],
        beta_schedule: str,
        linear_start: float,
        linear_end: float,
        cosine_s: float,
    ) -> None:
        if given_betas is not None:
            betas = given_betas
            timesteps = len(betas)
        else:
            args = beta_schedule, timesteps, linear_start, linear_end, cosine_s
            betas = make_beta_schedule(*args)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.t = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # cache for q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        one_m_cumprod = 1.0 - alphas_cumprod
        sqrt_one_m_cumprod = np.sqrt(one_m_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(sqrt_one_m_cumprod),
        )

        # cache for q(x_{t-1} | x_t, x_0)
        a0, a1 = self.v_posterior, 1.0 - self.v_posterior
        p0 = a0 * betas
        p1 = a1 * betas * (1.0 - alphas_cumprod_prev) / one_m_cumprod
        posterior_variance = p0 + p1
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer("posterior_coef1", to_torch(1.0 / np.sqrt(alphas)))
        self.register_buffer(
            "posterior_coef2",
            to_torch((1.0 - alphas) / sqrt_one_m_cumprod),
        )

        # TODO : check these!
        if self.parameterization == "eps":
            lvlb_weights = (
                0.5
                * self.betas**2
                / (
                    self.posterior_variance
                    * to_torch(alphas)
                    * (1.0 - self.alphas_cumprod)
                )
            )
        elif self.parameterization == "x0":
            lvlb_weights = to_torch(0.25 * np.sqrt(alphas_cumprod) / one_m_cumprod)
        else:
            msg = f"unrecognized parameterization '{self.parameterization}' occurred"
            raise NotImplementedError(msg)
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).any()


__all__ = [
    "DDPM",
]
