import numpy as np

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .ddpm import DDPM
from ..ae.kl import GaussianDistribution
from ..generator.vector_quantized import VQCodebookOut
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
        num_heads: Optional[int] = 8,
        num_head_channels: Optional[int] = None,
        use_spatial_transformer: bool = True,
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
        use_num_updates_in_ema: bool = True,
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
        sampler: str = "basic",
        sampler_config: Optional[Dict[str, Any]] = None,
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
            num_head_channels=num_head_channels,
            use_spatial_transformer=use_spatial_transformer,
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
            use_num_updates_in_ema=use_num_updates_in_ema,
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
            sampler=sampler,
            sampler_config=sampler_config,
        )
        first_stage_kw = first_stage_config or {}
        self.first_stage = freeze(DLZoo.load_model(first_stage, **first_stage_kw))
        self.scale_factor = first_stage_scale_factor
        # sanity check
        embedding_channels = self.first_stage.core.embedding_channels
        if in_channels != embedding_channels:
            raise ValueError(
                f"`in_channels` ({in_channels}) should be identical with the "
                f"`embedding_channels` ({embedding_channels}) of the "
                f"first_stage model ({first_stage})"
            )
        if out_channels != embedding_channels:
            raise ValueError(
                f"`out_channels` ({out_channels}) should be identical with the "
                f"`embedding_channels` ({embedding_channels}) of the "
                f"first_stage model ({first_stage})"
            )

    def decode(
        self,
        z: Tensor,
        *,
        cond: Optional[Any] = None,
        num_steps: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        latent = super().decode(
            z,
            cond=cond,
            num_steps=num_steps,
            verbose=verbose,
            **kwargs,
        )
        net = self._from_latent(latent)
        return net

    def reconstruct(
        self,
        net: Tensor,
        *,
        noise_steps: Optional[int] = None,
        cond: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tensor:
        latent = self._to_latent(net)
        kwargs.update(dict(noise_steps=noise_steps, cond=cond))
        net = super().reconstruct(latent, **kwargs)
        return net

    def _to_latent(self, net: Tensor) -> Tensor:
        net = self.first_stage.core.encode(net)
        if isinstance(net, GaussianDistribution):
            net = net.sample()
        elif isinstance(net, VQCodebookOut):
            net = net.z_q
        net = self.scale_factor * net
        return net

    def _from_latent(self, latent: Tensor) -> Tensor:
        latent = latent / self.scale_factor
        return self.first_stage.core.decode(latent)

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
            net = self._to_latent(net)
        return super()._get_input(net, cond)


__all__ = [
    "LDM",
]
