import torch

import torch.nn as nn

from torch import Tensor
from typing import Optional
from torch.nn import Module

from ..common import reuse_fn
from ..high_level import PreNorm
from ..attentions import Attention
from ..mixed_stacks import FeedForward


class PerceiverIO(Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_layers: int = 6,
        num_latents: int = 64,
        output_dim: Optional[int] = None,
        num_output: Optional[int] = None,
        latent_dim: int = 256,
        num_cross_heads: int = 1,
        num_latent_heads: int = 8,
        cross_latent_dim: Optional[int] = None,
        self_latent_dim: Optional[int] = None,
        feedforward_dropout: float = 0.0,
        feedforward_dim_ratio: float = 4.0,
        reuse_weights: bool = False,
        num_self_attn_repeat: int = 1,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = latent_dim
        feedforward_dim = int(round(latent_dim * feedforward_dim_ratio))

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            input_dim,
            module=Attention(
                latent_dim,
                num_cross_heads,
                k_dim=input_dim,
                embed_dim=cross_latent_dim or latent_dim,
            ),
        )
        get_cross_ff = lambda: PreNorm(
            latent_dim,
            module=FeedForward(
                latent_dim,
                latent_dim=feedforward_dim,
                dropout=feedforward_dropout,
            ),
        )
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            module=Attention(
                latent_dim,
                num_latent_heads,
                embed_dim=self_latent_dim or latent_dim,
                is_self_attention=True,
            ),
        )
        get_latent_ff = lambda: PreNorm(
            latent_dim,
            module=FeedForward(
                latent_dim,
                latent_dim=feedforward_dim,
                dropout=feedforward_dropout,
            ),
        )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            reuse_fn,
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff),
        )

        blocks = []
        for i in range(num_layers):
            cache_args = {"_cache": i > 0 and reuse_weights}
            self_attn_blocks = []
            for _ in range(num_self_attn_repeat):
                self_attn_blocks.append(
                    nn.ModuleList(
                        [
                            get_latent_attn(**cache_args),
                            get_latent_ff(**cache_args),
                        ]
                    )
                )
            blocks.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                        nn.ModuleList(self_attn_blocks),
                    ]
                )
            )
        self.layers = nn.ModuleList(blocks)

        self.decoder_cross_attn = PreNorm(
            output_dim,
            latent_dim,
            module=Attention(
                output_dim,
                num_cross_heads,
                k_dim=latent_dim,
                embed_dim=cross_latent_dim or latent_dim,
            ),
        )

        self.in_latent = nn.Parameter(torch.randn(num_latents, latent_dim))
        if num_output is None:
            self.out_latent = None
        else:
            self.out_latent = nn.Parameter(torch.randn(num_output, output_dim))

    def forward(
        self,
        net: Tensor,
        *,
        mask: Optional[Tensor] = None,
        out_queries: Optional[Tensor] = None,
    ) -> Tensor:
        b = net.shape[0]
        in_latent = torch.repeat_interleave(self.in_latent[None, ...], b, dim=0)
        for cross_attn, cross_ff, self_attn_blocks in self.layers:
            in_latent = cross_attn(in_latent, net, mask=mask) + in_latent
            in_latent = cross_ff(in_latent) + in_latent
            for self_attn, self_ff in self_attn_blocks:
                in_latent = self_attn(in_latent) + in_latent
                in_latent = self_ff(in_latent) + in_latent
        if self.out_latent is not None:
            out_queries = torch.repeat_interleave(self.out_latent[None, ...], b, dim=0)
        if out_queries is None:
            return in_latent
        return self.decoder_cross_attn(out_queries, in_latent)


__all__ = [
    "PerceiverIO",
]
