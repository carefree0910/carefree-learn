import torch

from typing import Any
from typing import Dict
from typing import Optional

from .constants import MASK_KEY
from ....types import tensor_dict_type
from ....protocol import TrainerState
from ....protocol import ModelProtocol
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....modules.blocks import MixedStackedEncoder


# TeT -> TextTransformer


@ModelProtocol.register("tet")
class TeTEncoder(ModelProtocol):
    def __init__(
        self,
        latent_dim: int = 384,
        context_length: int = 77,
        *,
        use_triu_attn_mask: bool = False,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        feedforward_kwargs: Optional[Dict[str, Any]] = None,
        use_positional_encoding: bool = True,
        norm_after_head: bool = False,
    ):
        super().__init__()
        if not use_triu_attn_mask:
            self.attention_mask = None
        else:
            mask = torch.empty(context_length, context_length, dtype=torch.bool)
            mask.fill_(True)
            mask.triu_(1)
            self.register_buffer("attention_mask", mask)
        if attention_kwargs is None:
            attention_kwargs = {}
        attention_kwargs.setdefault("bias", True)
        attention_kwargs.setdefault("num_heads", 6)
        self.encoder = MixedStackedEncoder(
            latent_dim,
            context_length,
            token_mixing_type="attention",
            token_mixing_config=attention_kwargs,
            channel_mixing_config=feedforward_kwargs,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            residual_after_norm=residual_after_norm,
            feedforward_dim_ratio=feedforward_dim_ratio,
            reduce_head=False,
            sequence_pool=False,
            use_head_token=False,
            use_positional_encoding=use_positional_encoding,
            norm_after_head=norm_after_head,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        mask = batch.get(MASK_KEY, self.attention_mask)
        net = self.encoder.pre_process(net, **kwargs)
        for block in self.encoder.mixing_blocks:
            net = block(net, mask=mask)
        net = self.encoder.post_process(net)
        return {LATENT_KEY: net}


__all__ = ["TeTEncoder"]
