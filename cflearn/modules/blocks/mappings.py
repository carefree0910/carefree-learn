import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Type
from typing import Optional
from torch.nn import Module
from cftool.misc import WithRegister

from .norms import BN
from .customs import Linear
from .activations import Activation


mapping_dict: Dict[str, Type["MappingBase"]] = {}


class MappingBase(Module, WithRegister["MappingBase"]):
    d = mapping_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()


@MappingBase.register("basic")
class Mapping(MappingBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.5,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        activation_config: Optional[Dict[str, Any]] = None,
        init_method: str = "xavier_normal",
        rank: Optional[int] = None,
        rank_ratio: Optional[float] = None,
    ):
        super().__init__()
        if bias is None:
            bias = not batch_norm
        self.linear = Linear(
            in_dim,
            out_dim,
            bias=bias,
            pruner_config=pruner_config,
            init_method=init_method,
            rank=rank,
            rank_ratio=rank_ratio,
        )
        self.bn = None if not batch_norm else BN(out_dim)
        if activation is None:
            self.activation: Optional[Module] = None
        else:
            self.activation = Activation.make(activation, activation_config)
        use_dropout = 0.0 < dropout < 1.0
        self.dropout = None if not use_dropout else nn.Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.linear.bias

    def forward(self, net: Tensor) -> Tensor:
        net = self.linear(net)
        if self.bn is not None:
            net = self.bn(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout is not None:
            net = self.dropout(net)
        return net

    @classmethod
    def simple(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: Optional[str] = None,
        pruner_config: Optional[Dict[str, Any]] = None,
    ) -> "Mapping":
        if activation != "glu":
            activation_config = {}
        else:
            activation_config = {"in_dim": out_dim, "bias": bias}
        return cls(
            in_dim,
            out_dim,
            bias=bias,
            pruner_config=pruner_config,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            activation_config=activation_config,
        )


@MappingBase.register("res")
class ResBlock(MappingBase):
    to_latent: Module

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        activation_config: Optional[Dict[str, Any]] = None,
        init_method: str = "xavier_normal",
    ):
        super().__init__()
        # input mapping
        if in_dim == out_dim:
            self.to_latent = nn.Identity()
        else:
            self.to_latent = Linear(
                in_dim,
                out_dim,
                bias=True if bias is None else bias,
                pruner_config=pruner_config,
                init_method=init_method,
            )
        # residual unit
        self.residual_unit = nn.Sequential(
            BN(out_dim) if batch_norm else nn.Identity(),
            Activation.make(activation, activation_config),
            nn.Identity() if not 0.0 < dropout < 1.0 else nn.Dropout(dropout),
            Linear(
                out_dim,
                out_dim,
                bias=True if bias is None else bias,
                pruner_config=pruner_config,
                init_method=init_method,
            ),
            nn.Identity() if not 0.0 < dropout < 1.0 else nn.Dropout(dropout),
        )

    def forward(self, net: Tensor) -> Tensor:
        net = self.to_latent(net)
        res = self.residual_unit(net)
        return net + res


@MappingBase.register("highway")
class HighwayBlock(MappingBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        init_method: str = "xavier_normal",
    ):
        super().__init__()
        self.linear_mapping = Linear(
            in_dim,
            out_dim,
            bias=True if bias is None else bias,
            pruner_config=pruner_config,
            init_method=init_method,
        )
        self.nonlinear_mapping = Mapping(
            in_dim,
            out_dim,
            bias=bias,
            pruner_config=pruner_config,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            init_method=init_method,
        )
        self.gate_linear = Linear(
            in_dim,
            out_dim,
            bias=True if bias is None else bias,
            pruner_config=pruner_config,
            init_method=init_method,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, net: Tensor) -> Tensor:
        linear = self.linear_mapping(net)
        nonlinear = self.nonlinear_mapping(net)
        gate = self.sigmoid(self.gate_linear(net))
        return gate * nonlinear + (1.0 - gate) * linear


__all__ = [
    "mapping_dict",
    "MappingBase",
]
