import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin
from cfdata.types import np_int_type

from .auxiliary import *
from ..misc.toolkit import *


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        pruner_config: Optional[Dict[str, Any]] = None,
        init_method: Optional[str] = "xavier_uniform",
        **kwargs: Any,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        if pruner_config is None:
            pruner = None
        else:
            pruner = Pruner(pruner_config, [out_dim, in_dim])
        self.config, self.pruner = shallow_copy_dict(kwargs), pruner
        self._use_bias, self._init_method = bias, init_method
        with torch.no_grad():
            self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if self.pruner is None:
            return self.linear(net)
        weight = self.pruner(self.linear.weight)
        return nn.functional.linear(net, weight, self.linear.bias)

    def reset_parameters(self) -> None:
        if self._init_method is None:
            return
        if self._init_method not in Initializer.defined_initialization:
            return
        initializer = Initializer(self.config.setdefault("initialize_config", {}))
        assert isinstance(self.linear.weight, nn.Parameter)
        initializer.initialize(self.linear.weight, self._init_method)
        bias_fill = self.config.setdefault("bias_fill", 0.0)
        if self._use_bias:
            self.linear.bias.data.fill_(bias_fill)


class Mapping(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.5,
        batch_norm: bool = True,
        activation: str = "ReLU",
        init_method: str = "xavier_uniform",
        **kwargs: Any,
    ):
        super().__init__()
        self.config = shallow_copy_dict(kwargs)
        if bias is None:
            bias = not batch_norm
        self.linear = Linear(
            in_dim,
            out_dim,
            bias=bias,
            pruner_config=pruner_config,
            init_method=init_method,
            **shallow_copy_dict(kwargs),
        )
        self.bn = None if not batch_norm else BN(out_dim)
        if activation is None:
            self.activation: Optional[nn.Module] = None
        else:
            activation_config = self.config.setdefault("activation_config", None)
            self.activation = Activations.make(activation, activation_config)
        use_dropout = 0.0 < dropout < 1.0
        self.dropout = None if not use_dropout else Dropout(dropout)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, net: torch.Tensor, *, reuse: bool = False) -> torch.Tensor:
        net = self.linear(net)
        if self.bn is not None:
            net = self.bn(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout is not None:
            net = self.dropout(net, reuse=reuse)
        return net


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int],
        num_units: List[int],
        mapping_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
        *,
        final_mapping_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        mappings: List[Union[Linear, Mapping]] = []
        if isinstance(mapping_configs, dict):
            mapping_configs = [mapping_configs] * len(num_units)
        for num_unit, mapping_config in zip(num_units, mapping_configs):
            mappings.append(Mapping(in_dim, num_unit, **mapping_config))
            in_dim = num_unit
        if out_dim is not None:
            if final_mapping_config is None:
                final_mapping_config = {}
            mappings.append(Linear(in_dim, out_dim, **final_mapping_config))
        self.mappings = nn.ModuleList(mappings)

    @property
    def weights(self) -> List[torch.Tensor]:
        return [mapping.weight for mapping in self.mappings]

    @property
    def biases(self) -> List[Optional[torch.Tensor]]:
        return [mapping.bias for mapping in self.mappings]

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        for mapping in self.mappings:
            net = mapping(net)
        return net

    @classmethod
    def simple(
        cls,
        in_dim: int,
        out_dim: Optional[int],
        num_units: List[int],
        *,
        bias: bool = False,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: Optional[str] = None,
    ) -> "MLP":
        mapping_config: Dict[str, Any]
        mapping_config = {"bias": bias, "dropout": dropout, "batch_norm": batch_norm}
        if activation is not None:
            mapping_config["activation"] = activation
        final_mapping_config = {"bias": bias}
        return cls(
            in_dim,
            out_dim,
            num_units,
            mapping_config,
            final_mapping_config=final_mapping_config,
        )


class DNDF(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_tree: int = 10,
        tree_depth: int = 4,
        is_regression: Optional[bool] = None,
        tree_proj_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._num_tree = num_tree
        self._tree_depth = tree_depth
        self._is_regression = out_dim == 1 if is_regression is None else is_regression
        self._num_leaf = 2 ** (self._tree_depth + 1)
        self._num_internals = self._num_leaf - 1
        self._output_dim = out_dim
        if tree_proj_config is None:
            tree_proj_config = {}
        tree_proj_config.setdefault("pruner_config", {})
        self.tree_proj = Linear(
            in_dim,
            self._num_internals * self._num_tree,
            **tree_proj_config,
        )
        self.leaves = nn.Parameter(
            torch.empty(self._num_tree, self._num_leaf, self._output_dim)
        )
        torch.nn.init.xavier_uniform_(self.leaves.data)
        # masks
        num_repeat, num_local_internals = self._num_leaf // 2, 1
        increment_masks = [
            torch.from_numpy(
                np.repeat([0, self._num_internals], num_repeat).astype(np_int_type)
            )
        ]
        for _ in range(1, self._tree_depth + 1):
            num_repeat //= 2
            num_local_internals *= 2
            increment_mask = np.repeat(
                np.arange(num_local_internals - 1, 2 * num_local_internals - 1), 2
            )
            increment_mask += np.tile([0, self._num_internals], num_local_internals)
            increment_mask = np.repeat(increment_mask, num_repeat)
            increment_mask = torch.from_numpy(increment_mask.astype(np_int_type))
            increment_masks.append(increment_mask)
        self.increment_masks: torch.Tensor
        self.register_buffer("tree_arange", torch.arange(num_tree)[..., None, None])
        self.register_buffer("increment_masks", torch.stack(increment_masks))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        num_batch = net.shape[0]
        tree_net = self.tree_proj(net)

        shape = num_batch, -1, self._num_internals
        p_left = torch.sigmoid(tree_net).view(*shape).transpose(0, 1)
        p_right = 1.0 - p_left
        flat_probabilities = torch.cat([p_left, p_right], dim=-1)
        flat_probabilities = flat_probabilities.contiguous().view(self._num_tree, -1)
        num_flat_prob = 2 * self._num_internals
        device = self.increment_masks.device
        batch_arange = torch.arange(0, num_flat_prob * num_batch, num_flat_prob)
        batch_indices = batch_arange.view(-1, 1).to(device)
        current_indices = batch_indices + self.increment_masks[0]
        flat_dim = flat_probabilities.shape[-1]
        tree_arange = self.tree_arange * flat_dim  # type: ignore
        routes = flat_probabilities.take(tree_arange + current_indices[None, ...])

        for i in range(1, self._tree_depth + 1):
            current_indices = batch_indices + self.increment_masks[i]
            current_indices = tree_arange + current_indices[None, ...]
            routes *= flat_probabilities.take(current_indices)
        features = routes.transpose(0, 1).contiguous().view(num_batch, -1)

        if self._is_regression or self._output_dim <= 1:
            leaves: Union[torch.Tensor, nn.Parameter] = self.leaves
        else:
            leaves = nn.functional.softmax(self.leaves, dim=-1)
        leaves = leaves.view(self._num_tree * self._num_leaf, self._output_dim)
        return features.matmul(leaves) / self._num_tree

    def reset_parameters(self) -> None:
        self.tree_proj.reset_parameters()
        nn.init.xavier_uniform_(self.leaves.data)


class TreeResBlock(nn.Module):
    def __init__(self, dim: int, dndf_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if dndf_config is None:
            dndf_config = {}
        self.dim = float(dim)
        self.in_dndf = DNDF(dim, dim, **shallow_copy_dict(dndf_config))
        self.inner_dndf = DNDF(dim, dim, **shallow_copy_dict(dndf_config))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        res = self.in_dndf(net)
        res = self.dim * res - 1.0
        res = self.inner_dndf(res)
        res = self.dim * res - 1.0
        return net + res


class InvertibleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        transition: Optional[nn.Module] = None,
        enable_permutation: bool = True,
    ):
        if dim % 2 != 0:
            raise ValueError("`dim` should be divided by 2")
        super().__init__()
        h_dim = int(dim // 2)
        if transition is None:
            transition = nn.Identity()
        self.transition = transition
        self.enable_permutation = enable_permutation
        if enable_permutation:
            permute_indices = np.random.permutation(h_dim)
            inverse_indices = np.argsort(permute_indices)
            permute_indices = to_torch(permute_indices).to(torch.long)
            inverse_indices = to_torch(inverse_indices).to(torch.long)
            self.register_buffer("permute_indices", permute_indices)
            self.register_buffer("inverse_indices", inverse_indices)

    def forward(
        self,
        net1: torch.Tensor,
        net2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        net1 = net1 + self.transition(net2)
        if self.enable_permutation:
            net2 = net2[..., self.permute_indices]
        return net2, net1

    def inverse(
        self,
        net1: torch.Tensor,
        net2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_permutation:
            net1 = net1[..., self.inverse_indices]
        net2 = net2 - self.transition(net1)
        return net2, net1


class ResInvertibleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        transition_builder: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()

        def get_block() -> InvertibleBlock:
            if transition_builder is None:
                transition = None
            else:
                transition = transition_builder(dim)
            return InvertibleBlock(
                dim,
                transition=transition,
                enable_permutation=False,
            )

        self.block1 = get_block()
        self.block2 = get_block()

    def forward(
        self,
        net1: torch.Tensor,
        net2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block2(*self.block1(net1, net2))

    def inverse(
        self,
        net1: torch.Tensor,
        net2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block1.inverse(*self.block2.inverse(net1, net2))


class PseudoInvertibleBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        in_activation: Optional[nn.Module] = None,
        inverse_in_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        dim = max(in_dim, out_dim)
        self.to_latent = Linear(in_dim, dim, bias=False)
        self.from_latent = MLP.simple(dim, in_dim, [dim, dim], activation="ReLU")
        msg = "`in_activation` and `inverse_in_activation` should be provided together"
        if in_activation is not None and inverse_in_activation is None:
            raise ValueError(msg)
        if in_activation is None and inverse_in_activation is not None:
            raise ValueError(msg)
        self.in_activation = in_activation
        self.inverse_in_activation = inverse_in_activation

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if self.in_activation is not None:
            net = self.in_activation(net)
        return self.to_latent(net)

    def inverse(self, net: torch.Tensor) -> torch.Tensor:
        net = self.from_latent(net)
        if self.inverse_in_activation is not None:
            net = self.inverse_in_activation(net)
        return net


class AttentionOutput(NamedTuple):
    output: torch.Tensor
    weights: torch.Tensor


class Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        *,
        dropout: float = 0.0,
        is_self_attention: bool = False,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        activation: Optional[str] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        q_linear_config: Optional[Dict[str, Any]] = None,
        k_linear_config: Optional[Dict[str, Any]] = None,
        v_linear_config: Optional[Dict[str, Any]] = None,
        in_linear_config: Optional[Dict[str, Any]] = None,
        out_linear_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.is_self_attn = is_self_attention
        if not is_self_attention:
            self.k_dim = k_dim if k_dim is not None else input_dim
            self.v_dim = v_dim if v_dim is not None else input_dim
        else:
            if k_dim is not None and k_dim != input_dim:
                raise ValueError("self attention is used but `k_dim` != `input_dim`")
            if v_dim is not None and v_dim != input_dim:
                raise ValueError("self attention is used but `v_dim` != `input_dim`")
            self.k_dim = self.v_dim = input_dim
        if embed_dim is None:
            embed_dim = min(32, input_dim) * num_heads
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("`embed_dim` must be divisible by `num_heads`")

        def _warn(prefix: str) -> None:
            msg = (
                f"self attention is used so `{prefix}_linear_config` will be ignored, "
                "please use `in_linear_config` instead"
            )
            print(f"{LoggingMixin.warning_prefix}{msg}")

        if q_linear_config is None:
            q_linear_config = {}
        elif is_self_attention:
            _warn("q")
        if k_linear_config is None:
            k_linear_config = {}
        elif is_self_attention:
            _warn("k")
        if v_linear_config is None:
            v_linear_config = {}
        elif is_self_attention:
            _warn("v")

        if is_self_attention:
            if in_linear_config is None:
                in_linear_config = {}
            self.in_linear = Linear(input_dim, 3 * self.embed_dim, **in_linear_config)
        else:
            self.q_linear = Linear(input_dim, self.embed_dim, **q_linear_config)
            self.k_linear = Linear(self.k_dim, self.embed_dim, **k_linear_config)
            self.v_linear = Linear(self.v_dim, self.embed_dim, **v_linear_config)

        if out_linear_config is None:
            out_linear_config = {}
        self.out_linear = Linear(self.embed_dim, input_dim, **out_linear_config)

        self.dropout = dropout
        self.activation = Activations.make(activation, activation_config)

    def _to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, in_feature = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> AttentionOutput:
        # `mask` represents slots which will be zeroed
        k_len = k.shape[1]
        if self.is_self_attn:
            q, k, v = self.in_linear(q).chunk(3, dim=-1)
        else:
            # B, Sq, Din -> B, Sq, D
            q = self.q_linear(q)
            # B, Sk, Dk -> B, Sk, D
            k = self.k_linear(k)
            # B, Sv, Dv -> B, Sk, D
            v = self.v_linear(v)
        q, k, v = map(self.activation, [q, k, v])
        # scale
        q = q * self.scaling
        # B, S*, D -> B * N_head, S*, D_head
        q, k, v = map(self._to_heads, [q, k, v])
        if mask is not None:
            # B, Sq, Sk -> B * N_head, Sq, Sk
            mask = mask.repeat(self.num_heads, 1, 1)
        # B * N_head, Sq, Sk
        raw_weights = torch.bmm(q, k.transpose(-2, -1))
        if mask is not None:
            raw_weights.masked_fill_(mask, float("-inf"))
        # B * N_head, Sq, Sk -> # B * N_head, Sq, Sk
        weights = F.softmax(raw_weights, dim=-1)
        if 0.0 < self.dropout < 1.0:
            weights = F.dropout(weights, self.dropout, self.training)
        # B * N_head, Sq, D_head
        output = torch.bmm(weights, v)
        # B * N_head, Sq, D_head -> B, N_head, Sq, D_head
        nb, q_len, d_head = output.shape
        output = output.view(nb // self.num_heads, self.num_heads, q_len, d_head)
        # B, N_head, Sq, D_head -> B, Sq, D
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(-1, q_len, self.embed_dim)
        # B, Sq, D -> B, Sq, Din
        output = self.activation(self.out_linear(output))
        return AttentionOutput(output, weights.view(-1, self.num_heads, q_len, k_len))


__all__ = [
    "Linear",
    "Mapping",
    "MLP",
    "DNDF",
    "TreeResBlock",
    "InvertibleBlock",
    "ResInvertibleBlock",
    "PseudoInvertibleBlock",
    "Attention",
]
