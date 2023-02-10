import torch

from typing import Any
from typing import Dict
from typing import Optional
from cftool.types import tensor_dict_type

from .constants import MASK_KEY
from ....schema import TrainerState
from ....schema import IDLModel
from ....constants import INPUT_KEY
from ....constants import LATENT_KEY
from ....modules.blocks import MixedStackedEncoder


# TeT -> TextTransformer


@IDLModel.register("tet")
class TeTEncoder(IDLModel):
    def __init__(
        self,
        latent_dim: int = 384,
        context_length: int = 77,
        *,
        use_triu_attn_mask: bool = False,
        num_layers: int = 12,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_position: str = "pre_norm",
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        embedding_norm: Optional[torch.nn.Module] = None,
        embedding_dropout: Optional[float] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 4.0,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        feedforward_kwargs: Optional[Dict[str, Any]] = None,
        use_positional_encoding: bool = True,
        head_pooler: Optional[str] = None,
        no_head_norm: Optional[bool] = None,
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
            norm_position=norm_position,
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
            embedding_norm=embedding_norm,
            embedding_dropout=embedding_dropout,
            residual_after_norm=residual_after_norm,
            latent_dim_ratio=feedforward_dim_ratio,
            use_head_token=False,
            head_pooler=head_pooler,
            use_positional_encoding=use_positional_encoding,
            is_vision_positional_encoding=False,
            no_head_norm=no_head_norm,
            norm_after_head=norm_after_head,
        )

    def forward(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        *,
        apply_head: bool = True,
        clip_skip: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        net = batch[INPUT_KEY]
        mask = batch.get(MASK_KEY, self.attention_mask)
        if mask is not None:
            t = net.shape[1]
            if t != mask.shape[0]:
                mask = mask[:t, :t]
        net = self.encoder.pre_process(net, **kwargs)
        for i, block in enumerate(self.encoder.mixing_blocks):
            net = block(net, mask=mask)
            if i == len(self.encoder.mixing_blocks) - 1 - clip_skip:
                break
        if apply_head:
            net = self.encoder.post_process(net)
        return {LATENT_KEY: net}


__all__ = ["TeTEncoder"]
