import json

import numpy as np

from enum import Enum
from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.misc import print_info
from cftool.array import tensor_dict_type

from .ddpm import make_condition_model
from .ddpm import DDPM
from .utils import CROSS_ATTN_TYPE
from ..ae.kl import GaussianDistribution
from ....zoo import DLZoo
from ....schema import IDLModel
from ....misc.toolkit import freeze
from ....misc.toolkit import get_tensors
from ....misc.toolkit import download_static
from ....modules.blocks import IHook
from ....modules.blocks import LoRAManager


@IDLModel.register("ldm")
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
        use_spatial_transformer: bool = False,
        num_transformer_layers: int = 1,
        context_dim: Optional[int] = None,
        signal_dim: int = 2,
        start_channels: int = 320,
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
        attn_split_chunk: Optional[int] = None,
        # first stage
        first_stage: str,
        first_stage_config: Optional[Dict[str, Any]] = None,
        first_stage_scale_factor: float = 1.0,
        # diffusion
        ema_decay: Optional[float] = None,
        use_num_updates_in_ema: bool = True,
        parameterization: str = "eps",
        ## condition
        condition_type: str = CROSS_ATTN_TYPE,
        condition_model: Optional[str] = None,
        use_first_stage_as_condition: bool = False,
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
            resample_with_resblock=resample_with_resblock,
            use_scale_shift_norm=use_scale_shift_norm,
            num_classes=num_classes,
            use_linear_in_transformer=use_linear_in_transformer,
            use_checkpoint=use_checkpoint,
            attn_split_chunk=attn_split_chunk,
            ema_decay=ema_decay,
            use_num_updates_in_ema=use_num_updates_in_ema,
            parameterization=parameterization,
            condition_type=condition_type,
            condition_model=condition_model,
            condition_config=condition_config,
            condition_learnable=condition_learnable,
            use_pretrained_condition=use_pretrained_condition,
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
        m = DLZoo.load_pipeline(first_stage, **first_stage_kw)
        self.first_stage = freeze(m.build_model.model)
        self.scale_factor = first_stage_scale_factor
        # condition
        if use_first_stage_as_condition:
            # avoid recording duplicate parameters
            self.condition_model = [make_condition_model(first_stage, self.first_stage)]
        # sanity check
        embedding_channels = self.first_stage.embedding_channels
        if in_channels != embedding_channels and condition_type == CROSS_ATTN_TYPE:
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
        start_step: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        latent = super().decode(
            z,
            cond=cond,
            num_steps=num_steps,
            start_step=start_step,
            verbose=verbose,
            **kwargs,
        )
        net = self._from_latent(latent)
        return net

    def _preprocess(self, net: Tensor, *, deterministic: bool = False) -> Tensor:
        net = self.first_stage.encode(net)
        if isinstance(net, GaussianDistribution):
            net = net.mode() if deterministic else net.sample()
        net = self.scale_factor * net
        return net

    def _from_latent(self, latent: Tensor) -> Tensor:
        latent = latent / self.scale_factor
        return self.first_stage.decode(latent, resize=False)

    def _get_cond(self, cond: Any) -> Tensor:
        if isinstance(self.condition_model, list):
            return self.condition_model[0](cond)
        return super()._get_cond(cond)


class SDLoRAMode(str, Enum):
    UNET = "unet"
    UNET_EXTENDED = "unet_extended"


def convert_lora(inp: Union[str, tensor_dict_type]) -> tensor_dict_type:
    inp = get_tensors(inp)
    with open(download_static("sd_lora_mapping", extension="json"), "r") as f:
        mapping = json.load(f)
    return {mapping[k]: v for k, v in inp.items()}


@IDLModel.register("sd")
class StableDiffusion(LDM):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lora_manager = LoRAManager()

    @property
    def has_lora(self) -> bool:
        return self.lora_manager.injected

    def load_lora(self, key: str, *, path: str) -> None:
        print_info(f"loading '{key}' from '{path}'")
        d = get_tensors(path)
        wk = "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"
        rank = d[wk].shape[0]
        mode = SDLoRAMode.UNET if "res" not in d else SDLoRAMode.UNET_EXTENDED
        inject_text_encoder = "lora_te_text_model_encoder_layers_0_mlp_fc1.alpha" in d
        print_info(
            f"preparing lora (rank={rank}, mode={mode}, "
            f"inject_text_encoder={inject_text_encoder})"
        )
        self.prepare_lora(key, rank, mode=mode, inject_text_encoder=inject_text_encoder)
        print_info("loading weights")
        self.lora_manager.load_pack_with(key, convert_lora(d))
        print_info(f"finished loading '{key}'")

    def prepare_lora(
        self,
        key: str,
        rank: int,
        *,
        mode: SDLoRAMode = SDLoRAMode.UNET,
        inject_text_encoder: bool = True,
    ) -> None:
        target_ancestors = {"SpatialTransformer", "GEGLU"}
        if mode == SDLoRAMode.UNET_EXTENDED:
            target_ancestors.add("ResBlock")
        if inject_text_encoder:
            target_ancestors.add("TeTEncoder")
        self.lora_manager.prepare(
            self,
            key=key,
            rank=rank,
            target_ancestors=target_ancestors,
        )

    def inject_lora(self, *keys: str) -> None:
        self.lora_manager.inject(self, *keys)

    def cleanup_lora(self) -> None:
        self.lora_manager.cleanup(self)

    def set_lora_scales(self, scales: Dict[str, float]) -> None:
        self.lora_manager.set_scales(scales)

    def get_lora_checkpoints(self) -> Dict[str, Optional[IHook]]:
        return self.lora_manager.checkpoint(self)

    def restore_lora_from(self, hooks: Dict[str, Optional[IHook]]) -> None:
        return self.lora_manager.restore(self, hooks)


__all__ = [
    "LDM",
    "SDLoRAMode",
    "StableDiffusion",
    "convert_lora",
]
