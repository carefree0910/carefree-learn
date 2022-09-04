import numpy as np

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .ddpm import DDPM
from ..ae.kl import GaussianDistribution
from ....zoo import DLZoo
from ....protocol import tensor_dict_type
from ....misc.toolkit import freeze
from ....misc.internal_ import register_custom_module


@register_custom_module("ldm")
class LDM(DDPM):
    def __init__(
        self,
        img_size: int,
        # unet
        in_channels: int,
        out_channels: int,
        *,
        num_heads: int = 8,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        start_channels: int = 320,
        num_res_blocks: int = 2,
        attention_downsample_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        resample_with_conv: bool = True,
        use_scale_shift_norm: bool = False,
        num_classes: Optional[int] = None,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        # first stage
        first_stage: str,
        first_stage_config: Optional[Dict[str, Any]] = None,
        first_stage_scale_factor: float = 1.0,
        # diffusion
        ema_decay: Optional[float] = None,
        parameterization: str = "eps",
        ## condition
        condition_type: str = "cross_attn",
        condition_model: Optional[str] = None,
        use_first_stage_as_condition: bool = False,
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
        default_start_T: int = 1000,
    ):
        self.use_first_stage_as_condition = use_first_stage_as_condition
        if use_first_stage_as_condition:
            if condition_learnable:
                raise ValueError(
                    "should not use ae as condition model "
                    "when `condition_learnable` is set to True"
                )
            condition_model = None
        super().__init__(
            img_size,
            in_channels,
            out_channels,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            context_dim=context_dim,
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
            ema_decay=ema_decay,
            parameterization=parameterization,
            condition_type=condition_type,
            condition_model=condition_model,
            condition_config=condition_config,
            condition_learnable=condition_learnable,
            v_posterior=v_posterior,
            timesteps=timesteps,
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
            loss_type=loss_type,
            l_simple_weight=l_simple_weight,
            original_elbo_weight=original_elbo_weight,
            learn_log_var=learn_log_var,
            log_var_init=log_var_init,
            default_start_T=default_start_T,
        )
        first_stage_kw = first_stage_config or {}
        first_stage_kw.setdefault("report", False)
        self.first_stage = freeze(DLZoo.load_model(first_stage, **first_stage_kw))
        self.scale_factor = first_stage_scale_factor

    def decode(
        self,
        z: Tensor,
        *,
        cond: Optional[Any] = None,
        start_T: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        temperature: float = 1.0,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        net = super().decode(
            z,
            cond=cond,
            start_T=start_T,
            num_timesteps=num_timesteps,
            temperature=temperature,
            verbose=verbose,
            **kwargs,
        )
        net = self.first_stage.core.decode(net)
        return net

    def _get_cond(self, cond: Any) -> Tensor:
        if self.condition_model is None:
            msg = "should not call `get_cond` when `condition_model` is not provided"
            raise ValueError(msg)
        encode_fn = getattr(self.condition_model, "encode", None)
        if encode_fn is None or not callable(encode_fn):
            cond = self.condition_model(cond)
        else:
            cond = encode_fn(cond)
            if isinstance(cond, GaussianDistribution):
                cond = cond.mode()
        return cond

    def _get_input(
        self,
        net: Tensor,
        cond: Optional[Tensor],
        *,
        in_decode: bool = False,
    ) -> Tuple[Tensor, tensor_dict_type]:
        if not in_decode:
            net = self.first_stage.core.encode(net)
            if isinstance(net, GaussianDistribution):
                net = net.sample()
            net = self.scale_factor * net
        return super()._get_input(net, cond)


__all__ = [
    "LDM",
]
