import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from itertools import combinations

from .base import MLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings
from ...misc.toolkit import eval_context
from ...modules.blocks import Activation


class SeparableMapping(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_groups: int,
        *,
        bias: bool = True,
        activation: Optional[str] = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        in_dim *= num_groups
        out_dim *= num_groups
        self.conv = nn.Conv1d(in_dim, out_dim, 1, groups=num_groups, bias=bias)
        self.bn = None if not batch_norm else nn.BatchNorm1d(out_dim)
        self.activation = Activation.make(activation)
        use_dropout = 0.0 < dropout < 1.0
        self.dropout = None if not use_dropout else nn.Dropout(dropout)

    # [B, num_groups * in_dim, D] -> [B, num_groups * out_dim, D]
    # note that they should be originated from [B, num_groups, in_dim / out_dim, D]
    def forward(self, net: Tensor) -> Tensor:
        net = self.conv(net)
        if self.bn is not None:
            net = self.bn(net)
        net = self.activation(net)
        if self.dropout is not None:
            net = self.dropout(net)
        return net


class NBMBlock(nn.Module):
    def __init__(
        self,
        order: int,
        num_bases: int,
        *,
        hidden_units: List[int],
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.order = order
        self.num_bases = num_bases

        blocks: List[nn.Module] = []
        in_dim = order
        for hidden_unit in hidden_units:
            mapping = SeparableMapping(
                in_dim,
                hidden_unit,
                num_bases,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
            blocks.append(mapping)
            in_dim = hidden_unit
        blocks.append(
            SeparableMapping(
                in_dim,
                1,
                num_bases,
                bias=bias,
                activation=None,
                batch_norm=False,
                dropout=0.0,
            )
        )

        self.net = nn.Sequential(*blocks)

    def forward(self, net: Tensor, indices: List[Tuple[int, ...]]) -> Tensor:
        for i, pack in enumerate(indices):
            if len(pack) != self.order:
                raise ValueError(
                    f"length of every tuple in indices must be {self.order}, "
                    f"but got {pack} at index {i}"
                )
        B = len(net)
        D = len(indices)
        # [B, D, order]
        net = net[..., indices]
        # [B, order, D]
        net = net.transpose(1, 2)
        # [B, 1, order, D]
        net = net.reshape(B, 1, self.order, D)
        # [B, num_bases, order, D]
        net = net.repeat_interleave(self.num_bases, 1)
        # [B, num_bases * order, D]
        net = net.contiguous().view(B, self.num_bases * self.order, D)
        # [B, num_bases, D]
        net = self.net(net)
        # [B, D, num_bases]
        return net.transpose(1, 2).contiguous()


@MLModel.register("nbm")
class NBM(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        nary: Optional[Dict[str, List[Tuple[int, ...]]]] = None,
        num_bases: int = 64,
        bases_dropout: float = 0.0,
        hidden_units: Optional[List[int]] = None,
        bias: bool = True,
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropout: float = 0.0,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        input_dim *= num_history
        if hidden_units is None:
            dim = max(32, min(1024, 2 * input_dim))
            hidden_units = 2 * [dim]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.bases = nn.ModuleDict()
        self.nary = nary or {"1": list(combinations(range(input_dim), 1))}
        self.num_bases = num_bases
        for order in self.nary:
            self.bases[order] = NBMBlock(
                int(order),
                num_bases,
                hidden_units=hidden_units,
                bias=bias,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
        self.bases_dropout = nn.Dropout(p=bases_dropout)

        self.all_indices = []
        for indices in self.nary.values():
            self.all_indices += indices
        self.latent_dim = len(self.all_indices)

        self.featurizer = nn.Conv1d(
            self.latent_dim * num_bases,
            self.latent_dim,
            kernel_size=1,
            groups=self.latent_dim,
            bias=bias,
        )
        self.head = nn.Linear(self.latent_dim, output_dim, bias=True)

    def get_key(self, order: str) -> str:
        return f"order_{order}"

    def forward(self, net: Tensor, *, return_features: bool = False) -> Tensor:
        B = len(net)
        if len(net.shape) > 2:
            net = net.contiguous().view(B, -1)
        bases = []
        for order, indices in self.nary.items():
            base = self.bases[order](net, indices)
            base = self.bases_dropout(base)
            bases.append(base)
        # [B, latent_dim, num_bases]
        net = torch.cat(bases, dim=1)
        # [B, latent_dim * num_bases, 1]
        net = net.view(B, self.latent_dim * self.num_bases, 1)
        # [B, latent_dim]
        net = self.featurizer(net).squeeze(2)
        if return_features:
            return net
        # [B, out_dim]
        net = self.head(net)
        return net

    def inspect(
        self,
        net: Tensor,
        x_dims: Tuple[int, ...],
        y_dim: int,
        *,
        already_extracted: bool = False,
    ) -> Tensor:
        try:
            idx = self.all_indices.index(x_dims)
        except ValueError:
            raise ValueError(f"`x_dims` ({x_dims}) is not in the pre-defined nary")
        with eval_context(self):
            B = len(net)
            if len(net.shape) > 2:
                net = net.contiguous().view(B, -1)
            order = len(x_dims)
            # [B, 1, num_bases]
            indices = [tuple(range(order))] if already_extracted else [x_dims]
            net = self.bases[str(order)](net, indices)
            net = self.bases_dropout(net)
            # [B, num_bases]
            net = net.view(B, self.num_bases, 1).squeeze(2)
            # [B, 1]
            w = self.featurizer.weight[[idx]].squeeze(2)
            b = None if not self.bias else self.featurizer.bias[[idx]]
            net = F.linear(net, w, b)
            # [B, 1]
            w = self.head.weight[..., [idx]][[y_dim]]
            net = F.linear(net, w)
        return net


__all__ = ["NBM"]
