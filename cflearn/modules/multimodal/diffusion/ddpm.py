import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import ContextManager
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.array import to_torch
from cftool.types import tensor_dict_type

from .unet import ControlNet
from .unet import UNetDiffuser
from .utils import cond_type
from .utils import extract_to
from .utils import get_timesteps
from .utils import ADM_KEY
from .utils import ADM_TYPE
from .utils import CONCAT_KEY
from .utils import CONCAT_TYPE
from .utils import HYBRID_TYPE
from .utils import CROSS_ATTN_KEY
from .utils import CROSS_ATTN_TYPE
from .utils import CONTROL_HINT_KEY
from .utils import CONTROL_HINT_END_KEY
from .utils import CONTROL_HINT_START_KEY
from .samplers import is_misc_key
from .samplers import ISampler
from .samplers import DDPMQSampler
from .cond_models import condition_models
from .cond_models import specialized_condition_models
from ...cv import register_generator
from ...cv import IGenerator
from ...cv import DecoderInputs
from ...common import EMA
from ....schema import device_type
from ....toolkit import to_eval
from ....toolkit import get_dtype
from ....toolkit import get_device
from ....constants import PREDICTIONS_KEY
from ....zoo.common import load_module


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
        timestep_array = arange / timesteps + cosine_s
        alphas = timestep_array / (1 + cosine_s) * np.pi / 2
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


def make_condition_model(key: str, m: nn.Module) -> nn.Module:
    if specialized_condition_models.has(key):
        return m
    condition_base = condition_models.get(key)
    if condition_base is None:
        return m
    return condition_base(m)


@register_generator("ddpm")
class DDPM(IGenerator):
    cond_key = "cond"
    noise_key = "noise"
    timesteps_key = "timesteps"
    identity_condition_model = "identity"

    sampler: ISampler
    control_model: Optional[Union[ControlNet, nn.ModuleDict]]
    control_model_lazy: bool
    control_scales: Optional[Union[List[float], List[List[float]]]]

    # buffers

    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    posterior_variance: Tensor
    posterior_log_variance_clipped: Tensor
    posterior_coef1: Tensor
    posterior_coef2: Tensor
    lvlb_weights: Tensor

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
        use_spatial_transformer: bool = False,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        resample_with_conv: bool = True,
        resample_with_resblock: bool = False,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_linear_in_transformer: bool = False,
        use_checkpoint: bool = False,
        hooks_kwargs: Optional[Dict[str, Any]] = None,
        # ControlNet
        only_mid_control: bool = False,
        # diffusion
        ema_decay: Optional[float] = None,
        use_num_updates_in_ema: bool = True,
        parameterization: str = "eps",
        ## condition
        condition_type: str = CROSS_ATTN_TYPE,
        condition_model: Optional[str] = None,
        condition_config: Optional[Dict[str, Any]] = None,
        condition_learnable: bool = False,
        use_pretrained_condition: bool = False,
        ## noise schedule
        v_posterior: float = 0.0,
        timesteps: int = 1000,
        given_betas: Optional[np.ndarray] = None,
        beta_schedule: str = "linear",
        linear_start: float = 1.0e-4,
        linear_end: float = 2.0e-2,
        cosine_s: float = 8.0e-3,
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
        self.unet_kw = dict(
            in_channels=in_channels,
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
            resample_with_resblock=resample_with_resblock,
            use_scale_shift_norm=use_scale_shift_norm,
            num_classes=num_classes,
            use_linear_in_transformer=use_linear_in_transformer,
            use_checkpoint=use_checkpoint,
            hooks_kwargs=hooks_kwargs,
        )
        self.unet = UNetDiffuser(out_channels=out_channels, **self.unet_kw)  # type: ignore
        # ControlNet
        self.control_model = None
        self.control_model_lazy = False
        self.only_mid_control = only_mid_control
        self.num_control_scales = len(self.unet.output_blocks) + 1
        self.control_scales = None
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
        self.use_pretrained_condition = use_pretrained_condition
        self._initialize_condition_model(
            condition_model,
            condition_config,
            condition_learnable,
            use_pretrained_condition,
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
        self.learn_log_var = learn_log_var
        log_var = torch.full(fill_value=log_var_init, size=(self.t,))
        if not learn_log_var:
            self.log_var = log_var
        else:
            self.log_var = nn.Parameter(log_var, requires_grad=True)
        # sampler
        self.switch_sampler(sampler, sampler_config)
        self.q_sampler = DDPMQSampler(self)
        self.q_sampler.reset_buffers(
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )

    def forward(self, net: Tensor, cond: Optional[Tensor]) -> tensor_dict_type:
        timesteps = torch.randint(0, self.t, (net.shape[0],), device=net.device).long()
        net = self.preprocess(net)
        noise = torch.randn_like(net)
        net = self.q_sampler.q_sample(net, timesteps, noise)
        unet_out = self.denoise(net, timesteps, cond, timesteps[0].item(), self.t)
        return {
            PREDICTIONS_KEY: unet_out,
            self.noise_key: noise,
            self.timesteps_key: timesteps,
        }

    # inherit

    def generate_z(self, num_samples: int) -> Tensor:
        shape = num_samples, self.in_channels, self.img_size, self.img_size
        return torch.randn(shape, dtype=get_dtype(self), device=get_device(self))

    def decode(self, inputs: DecoderInputs) -> Tensor:
        kwargs = shallow_copy_dict(inputs.kwargs or {})
        kwargs.setdefault("cond", inputs.cond)
        kwargs.setdefault("num_steps", inputs.num_steps)
        kwargs.setdefault("start_step", inputs.start_step)
        kwargs.setdefault("verbose", inputs.verbose)
        inputs.kwargs = kwargs
        with self._control_model_context():
            return self.sampler.sample(inputs.z, **kwargs)

    def sample(
        self,
        num_samples: int,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = None,
        clip_output: bool = True,
        verbose: bool = True,
        kwargs: Optional[Dict[str, Any]] = None,
        # compatible with the parent class
        **other_kw: Any,
    ) -> Tensor:
        kw = shallow_copy_dict(other_kw)
        kw["cond"] = cond
        kw["num_steps"] = num_steps
        kw["start_step"] = start_step
        kw["verbose"] = verbose
        if kwargs is not None:
            kw.update(kwargs)
        sampled = super().sample(num_samples, kwargs=kw)
        if clip_output:
            sampled = torch.clip(sampled, -1.0, 1.0)
        return sampled

    def reconstruct(
        self,
        net: Tensor,
        *,
        cond: Optional[Any] = None,
        noise_steps: Optional[int] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        # compatible with the parent class
        **other_kw: Any,
    ) -> Tensor:
        net = self.preprocess(net)
        if noise_steps is None:
            noise_steps = self.t
        ts = get_timesteps(noise_steps - 1, net.shape[0], net.device)
        z = self.q_sampler.q_sample(net, ts)
        kw = shallow_copy_dict(other_kw)
        if kwargs is not None:
            kw.update(kwargs)
        net = self.decode(DecoderInputs(z=z, cond=cond, kwargs=kw))
        return net

    # optional callbacks

    def get_cond(self, cond: Any) -> cond_type:
        if self.condition_model is None:
            msg = "should not call `get_cond` when `condition_model` is not provided"
            raise ValueError(msg)
        if not isinstance(cond, dict):
            return self.condition_model(cond)
        for k, v in cond.items():
            if not is_misc_key(k):
                cond[k] = self.condition_model(v)
        return cond

    def preprocess(self, net: Tensor, *, deterministic: bool = False) -> Tensor:
        return net

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

    def denoise(
        self,
        image: Tensor,
        timesteps: Tensor,
        cond: Optional[cond_type],
        step: int,
        total_step: int,
    ) -> Tensor:
        net = image
        cond_kw = {}
        cond_type = self.condition_type
        if cond is not None and cond_type != "none":
            if cond_type == CONCAT_TYPE:
                if isinstance(cond, dict):
                    cond = cond[CONCAT_KEY]
                net = torch.cat([net, cond], dim=1)
            elif cond_type == CROSS_ATTN_TYPE:
                if not isinstance(cond, dict):
                    cond_kw = {CROSS_ATTN_KEY: cond}
                else:
                    cond_kw = {CROSS_ATTN_KEY: cond[CROSS_ATTN_KEY]}
            elif cond_type == HYBRID_TYPE:
                if not isinstance(cond, dict):
                    raise ValueError("`cond` should be a dict when `hybrid` is applied")
                concat = cond[CONCAT_KEY].repeat_interleave(len(net), dim=0)
                net = torch.cat([net, concat], dim=1)
                cond_kw = {CROSS_ATTN_KEY: cond[CROSS_ATTN_KEY]}
            elif cond_type == ADM_TYPE:
                if isinstance(cond, dict):
                    cond_kw = cond
                else:
                    cond_kw = {ADM_KEY: cond}
            else:
                raise ValueError(f"unrecognized condition type {cond_type} occurred")
        if self.control_model is not None:
            if not isinstance(cond, dict):
                raise ValueError("`cond` should be a dict when `control_model` is used")
            hint = cond.get(CONTROL_HINT_KEY)
            hint_start = cond.get(CONTROL_HINT_START_KEY)
            hint_end = cond.get(CONTROL_HINT_END_KEY)
            check_hint_start = lambda start: start is None or start * total_step <= step
            check_hint_end = lambda end: end is None or end * total_step >= step
            if hint is None:
                raise ValueError("`hint` should be provided for `control_model`")
            if isinstance(self.control_model, ControlNet):
                if not isinstance(hint, Tensor):
                    raise ValueError("`hint` should be a Tensor for single control")
                if hint_start is not None and isinstance(hint_start, list):
                    raise ValueError("`hint_start` should be float for single control")
                if hint_end is not None and isinstance(hint_end, list):
                    raise ValueError("`hint_end` should be float for single control")
                if not check_hint_start(hint_start) or not check_hint_end(hint_end):
                    ctrl = None
                else:
                    ctrl = self.control_model(net, hint, timesteps=timesteps, **cond_kw)
                    if self.control_scales is None:
                        scales = [1.0] * self.num_control_scales
                    else:
                        if isinstance(self.control_scales[0], list):
                            raise ValueError("`control_scales` should be list of float")
                        scales = self.control_scales  # type: ignore
                    ctrl = [c * scale for c, scale in zip(ctrl, scales)]
            else:
                if not isinstance(hint, list):
                    raise ValueError("`hint` should be a list for control settings")
                if not isinstance(hint_start, list):
                    raise ValueError("`hint_start` should be a list of Optional[float]")
                if not isinstance(hint_end, list):
                    raise ValueError("`hint_end` should be a list of Optional[float]")
                target_keys = set(self.control_model.keys())
                hint_types = set(pair[0] for pair in hint)
                if hint_types - target_keys:
                    msg = f"`hint` ({hint_types}) should not exceed following keys: {', '.join(sorted(target_keys))}"
                    raise ValueError(msg)
                ctrl = [0.0] * self.num_control_scales
                any_activated = False
                for i, ((i_type, i_hint), i_start, i_end) in enumerate(
                    zip(hint, hint_start, hint_end)
                ):
                    if not check_hint_start(i_start) or not check_hint_end(i_end):
                        continue
                    any_activated = True
                    i_cmodel = self.control_model[i_type]
                    # inpainting workaround
                    if i_cmodel.in_channels == net.shape[1]:
                        cnet = net
                    else:
                        cnet = net[:, : i_cmodel.in_channels]
                    i_control = i_cmodel(cnet, i_hint, timesteps=timesteps, **cond_kw)
                    if self.control_scales is None:
                        i_scales = [1.0] * self.num_control_scales
                    else:
                        if not isinstance(self.control_scales[i], list):
                            raise ValueError("`control_scales` should be list of list")
                        i_scales = self.control_scales[i]  # type: ignore
                    for j, (j_c, j_scale) in enumerate(zip(i_control, i_scales)):
                        ctrl[j] += j_c * j_scale
                if not any_activated:
                    ctrl = None
            cond_kw["control"] = ctrl
            cond_kw["only_mid_control"] = self.only_mid_control
        return self.unet(net, timesteps=timesteps, **cond_kw)

    def predict_eps_from_z_and_v(
        self,
        image: Tensor,
        ts: Tensor,
        v: Tensor,
    ) -> Tensor:
        num_dim = image.ndim
        return (
            extract_to(self.sqrt_alphas_cumprod, ts, num_dim) * v
            + extract_to(self.sqrt_one_minus_alphas_cumprod, ts, num_dim) * image
        )

    def predict_start_from_z_and_v(
        self,
        image: Tensor,
        ts: Tensor,
        v: Tensor,
    ) -> Tensor:
        num_dim = image.ndim
        return (
            extract_to(self.sqrt_alphas_cumprod, ts, num_dim) * image
            - extract_to(self.sqrt_one_minus_alphas_cumprod, ts, num_dim) * v
        )

    def make_control_net(
        self,
        hint_channels: Union[int, Dict[str, int]],
        lazy: bool = False,
        *,
        target_key: Optional[str] = None,
    ) -> None:
        def _make(n: int) -> ControlNet:
            # temporarily make inpainting compatible
            kw = shallow_copy_dict(self.unet_kw)
            kw["in_channels"] = 4
            return ControlNet(hint_channels=n, **kw)  # type: ignore

        if target_key is not None:
            if not isinstance(self.control_model, nn.ModuleDict):
                msg = "`target_key` can only be used in multi `ControlNet`"
                raise ValueError(msg)
            if not isinstance(hint_channels, int):
                msg = "`hint_channels` should be int when `target_key` is used"
                raise ValueError(msg)
            if target_key not in self.control_model:
                self.control_model[target_key] = _make(hint_channels)
        else:
            if isinstance(hint_channels, int):
                self.control_model = _make(hint_channels)
            else:
                self.control_model = nn.ModuleDict(
                    {
                        key: _make(key_channels)
                        for key, key_channels in hint_channels.items()
                    }
                )

        self.control_model_lazy = lazy
        if not lazy:
            self._control_model_to()

    def rename_control_net(self, old: str, new: str) -> None:
        if not isinstance(self.control_model, nn.ModuleDict):
            msg = "`rename_control_net` is only available when multiple `ControlNet` are used"
            raise ValueError(msg)
        if old not in self.control_model:
            raise ValueError(f"cannot find '{old}' in current models")
        m = self.control_model.pop(old)
        self.control_model[new] = m

    def load_control_net_with(self, name: str, d: tensor_dict_type) -> None:
        if not isinstance(self.control_model, nn.ModuleDict):
            msg = "`load_control_net_with` is only available when multiple `ControlNet` are used"
            raise ValueError(msg)
        if name not in self.control_model:
            raise ValueError(f"cannot find '{name}' in current models")
        self.control_model[name].load_state_dict(d)

    def detach_control_net(self) -> None:
        if self.control_model is not None:
            self.control_model.to("cpu")
            del self.control_model
            self.control_model = None

    # internal

    def _control_model_to(self, device: device_type = None) -> None:
        if self.control_model is not None:
            p = list(self.unet.parameters())[0]
            if device is None:
                device = p.device
            dtype = p.dtype
            self.control_model.to(device, dtype=dtype)

    def _control_model_context(self) -> ContextManager:
        class _:
            def __init__(self, m: DDPM):
                self.m = m.control_model
                self.lazy = m.control_model_lazy
                self.to = m._control_model_to

            def __enter__(self) -> None:
                if self.m is not None and self.lazy:
                    self.to()

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                if self.m is not None and self.lazy:
                    self.to("cpu")
                    torch.cuda.empty_cache()

        return _(self)

    def _initialize_condition_model(
        self,
        condition_model: Optional[str],
        condition_config: Optional[Dict[str, Any]],
        condition_learnable: bool,
        use_pretrained_condition: bool,
    ) -> None:
        if condition_model is None:
            self.condition_model = None
            return
        if condition_model == self.identity_condition_model:
            self.condition_model = nn.Identity()
            return
        specialized = specialized_condition_models.get(condition_model)
        if specialized is not None:
            m = safe_execute(specialized, condition_config or {})
        else:
            m = load_module(
                condition_model,
                pretrained=use_pretrained_condition,
                **(condition_config or {}),
            )
        self.condition_model = make_condition_model(condition_model, m)
        if not condition_learnable:
            to_eval(self.condition_model)

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
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2
                / (
                    2
                    * self.posterior_variance
                    * to_torch(alphas)
                    * (1.0 - self.alphas_cumprod)
                )
            )
        else:
            msg = f"unrecognized parameterization '{self.parameterization}' occurred"
            raise NotImplementedError(msg)
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).any()


__all__ = [
    "DDPM",
]
