import copy
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import NamedTuple
from functools import wraps
from functools import partial
from torch.nn import Module
from torch.nn import ModuleList
from torch.fft import fft
from torch.autograd import Function
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import WithRegister
from cftool.array import squeeze
from torch.nn.modules.utils import _pair
from torch.nn.modules.pooling import _MaxUnpoolNd

from ..types import tensor_dict_type
from ..protocol import WithDeviceMixin
from ..misc.toolkit import filter_kw
from ..misc.toolkit import interpolate
from ..misc.toolkit import adain_with_params
from ..misc.toolkit import eval_context


# auxiliary


def get_clones(
    module: Module,
    n: int,
    *,
    return_list: bool = False,
) -> Union[ModuleList, List[Module]]:
    module_list = [module]
    for _ in range(n - 1):
        module_list.append(copy.deepcopy(module))
    if return_list:
        return module_list
    return ModuleList(module_list)


def reuse_fn(f: Callable) -> Callable:
    cache = None

    @wraps(f)
    def cached_fn(*args: Any, _cache: bool = True, **kwargs: Any) -> Callable:
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class Lambda(Module):
    def __init__(self, fn: Callable, name: str = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class BN(nn.BatchNorm1d):
    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


class LN(nn.LayerNorm):
    def forward(self, net: Tensor) -> Tensor:
        if len(net.shape) != 4 or len(self.normalized_shape) != 1:
            return super().forward(net)
        batch_size = net.shape[0]
        if batch_size == 1:
            mean = net.mean().view(1, 1, 1, 1)
            std = net.std().view(1, 1, 1, 1)
        else:
            mean = net.view(batch_size, -1).mean(1).view(batch_size, 1, 1, 1)
            std = net.view(batch_size, -1).std(1).view(batch_size, 1, 1, 1)
        net = (net - mean) / (std + self.eps)
        if self.elementwise_affine:
            w = self.weight.view(-1, 1, 1)
            b = self.bias.view(-1, 1, 1)
            net = net * w + b
        return net


class EMA(Module):
    def __init__(
        self,
        decay: float,
        named_parameters: List[Tuple[str, nn.Parameter]],
    ):
        super().__init__()
        self._decay = decay
        self._named_parameters = named_parameters
        for name, param in self.tgt_params:
            self.register_buffer(self.get_name(True, name), param.data.clone())
            self.register_buffer(self.get_name(False, name), param.data.clone())

    @staticmethod
    def get_name(train: bool, name: str) -> str:
        prefix = "tr" if train else "ema"
        return f"{prefix}_{name}"

    @property
    def tgt_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return map(
            lambda pair: (pair[0].replace(".", "_"), pair[1]),
            self._named_parameters,
        )

    def forward(self) -> None:
        for name, param in self.tgt_params:
            setattr(self, self.get_name(True, name), param.data.clone())
            ema_name = self.get_name(False, name)
            ema_attr = getattr(self, ema_name)
            ema = (1.0 - self._decay) * param.data + self._decay * ema_attr
            setattr(self, ema_name, ema.clone())

    def train(self, mode: bool = True) -> "EMA":
        super().train(mode)
        for name, param in self.tgt_params:
            param.data = getattr(self, self.get_name(mode, name)).clone()
        return self

    def extra_repr(self) -> str:
        max_str_len = max(len(name) for name, _ in self.tgt_params)
        return "\n".join(
            [f"(0): decay_rate={self._decay}\n(1): Params("]
            + [
                f"  {name:<{max_str_len}s} - Tensor({list(param.shape)})"
                for name, param in self.tgt_params
            ]
            + [")"]
        )


class MTL(Module):
    def __init__(
        self,
        num_tasks: int,
        method: Optional[str] = None,
    ):
        super().__init__()
        self._n_task, self._method = num_tasks, method
        if method is None or method == "naive":
            pass
        elif method == "softmax":
            self.w = torch.nn.Parameter(torch.ones(num_tasks))
        else:
            raise NotImplementedError(f"MTL method '{method}' not implemented")
        self.registered = False
        self._slice: Optional[int] = None
        self._registered: Dict[str, int] = {}

    def register(self, names: List[str]) -> None:
        if self.registered:
            raise ValueError("re-register is not permitted")
        self._rev_registered = {}
        for name in sorted(names):
            idx = len(self._registered)
            self._registered[name], self._rev_registered[idx] = idx, name
        self._slice, self.registered = len(names), True
        if self._slice > self._n_task:
            raise ValueError("registered names are more than n_task")

    def forward(
        self,
        loss_dict: tensor_dict_type,
        naive: bool = False,
    ) -> Tensor:
        if not self.registered:
            raise ValueError("losses need to be registered")
        if naive or self._method is None:
            return self._naive(loss_dict)
        return getattr(self, f"_{self._method}")(loss_dict)

    @staticmethod
    def _naive(loss_dict: tensor_dict_type) -> Tensor:
        return sum(loss_dict.values())  # type: ignore

    def _softmax(self, loss_dict: tensor_dict_type) -> Tensor:
        assert self._slice is not None
        w = self.w if self._slice == self._n_task else self.w[: self._slice]
        softmax_w = nn.functional.softmax(w, dim=0)
        losses = []
        for key, loss in loss_dict.items():
            idx = self._registered.get(key)
            losses.append(loss if idx is None else loss * softmax_w[idx])
        final_loss: Tensor = sum(losses)  # type: ignore
        return final_loss * self._slice

    def extra_repr(self) -> str:
        method = "naive" if self._method is None else self._method
        return f"n_task={self._n_task}, method='{method}'"


# activations


activations: Dict[str, Type["Activation"]] = {}


class Activation(WithRegister["Activation"], Module, metaclass=ABCMeta):
    d = activations

    def __init__(self, **kwargs: Any):
        super().__init__()

    @classmethod
    def make(
        cls,
        name: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Module:
        if name is None:
            return nn.Identity()
        if config is None:
            config = {}
        if name.startswith("leaky_relu"):
            splits = name.split("_")
            if len(splits) == 3:
                config["negative_slope"] = float(splits[-1])
            config.setdefault("inplace", True)
            return nn.LeakyReLU(**config)
        if name.lower() == "relu":
            name = "ReLU"
            config.setdefault("inplace", True)
        base = cls.d.get(name, getattr(nn, name, None))
        if base is not None:
            return base(**config)
        func = getattr(torch, name, getattr(F, name, None))
        if func is None:
            raise NotImplementedError(
                "neither pytorch nor custom Activation "
                f"implemented activation '{name}'"
            )
        return Lambda(partial(func, **config), name)


@Activation.register("glu")
class GLU(Activation):
    def __init__(self, *, in_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2 * in_dim, bias)

    def forward(self, net: Tensor) -> Tensor:
        projection, gate = self.linear(net).chunk(2, dim=1)
        return projection * torch.sigmoid(gate)


@Activation.register("mish")
class Mish(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net * (torch.tanh(F.softplus(net)))


@Activation.register("atanh")
class ATanh(Activation):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.atanh(torch.clamp(net, -1.0 + self.eps, 1.0 - self.eps))


@Activation.register("isoftplus")
class InverseSoftplus(Activation):
    def __init__(self, *, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, net: Tensor) -> Tensor:
        return torch.log(net.clamp_min(self.eps).exp() - 1.0)


@Activation.register("sign")
class Sign(Activation):
    def __init__(
        self,
        *,
        eps: float = 1.0e-12,
        randomize_at_zero: bool = False,
        differentiable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.differentiable = differentiable
        self.randomize_at_zero = randomize_at_zero

    def forward(self, net: Tensor) -> Tensor:
        if self.randomize_at_zero:
            net = net + (2 * torch.empty_like(net).uniform_() - 1.0) * self.eps
        sign = torch.sign(net)
        if not self.differentiable:
            return sign
        return net + (sign - net).detach()


@Activation.register("one_hot")
class OneHot(Activation):
    def __init__(self, *, differentiable: bool = True):
        super().__init__()
        self.differentiable = differentiable

    def forward(self, net: Tensor) -> Tensor:
        maxed = torch.max(net, dim=1, keepdim=True)[0]
        one_hot = net * (net == maxed).to(torch.float32)
        if not self.differentiable:
            return one_hot
        return net + (one_hot - net).detach()


@Activation.register("sine")
class Sine(Activation):
    def __init__(self, *, w: float = 1.0):
        super().__init__()
        self.w = w

    def forward(self, net: Tensor) -> Tensor:
        return torch.sin(self.w * net)


@Activation.register("h_swish")
class HSwish(Activation):
    def __init__(self, *, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, net: Tensor) -> Tensor:
        return net * (self.relu(net + 3.0) / 6.0)


@Activation.register("quick_gelu")
class QuickGELU(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net * torch.sigmoid(1.702 * net)


@Activation.register("diff_relu")
class DiffReLU(Activation):
    def forward(self, net: Tensor) -> Tensor:
        return net + (torch.relu(net) - net).detach()


# custom blocks


class LeafAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Tensor:
        net, leaves = args
        softmax_leaves = F.softmax(leaves, dim=1)
        ctx.save_for_backward(net, softmax_leaves.t())
        return net.mm(softmax_leaves)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[Tensor], ...]:
        grad_output = grad_outputs[0]
        if grad_output is None:
            return None, None
        net, softmax_leaves = ctx.saved_tensors
        net_grad = grad_output.mm(softmax_leaves)
        sub_grad = grad_output.t().mm(net)
        sub_grad2 = (softmax_leaves * sub_grad).sum(0, keepdim=True)
        leaves_grad = softmax_leaves * (sub_grad - sub_grad2)
        return net_grad, leaves_grad.t()


class Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Tensor:
        (
            net,
            tree_arange,
            batch_indices,
            ones,
            increment_masks,
            num_tree,
            num_batch,
            tree_depth,
            num_internals,
        ) = args
        shape = num_batch, -1, num_internals
        sigmoid_net = torch.sigmoid(net)
        p_left = sigmoid_net.view(*shape).transpose(0, 1)
        p_right = 1.0 - p_left
        flat_probabilities = torch.cat([p_left, p_right], dim=-1)
        flat_probabilities = flat_probabilities.contiguous().view(num_tree, -1)
        current_indices = batch_indices + increment_masks[0]
        flat_dim = flat_probabilities.shape[-1]
        tree_arange = tree_arange * flat_dim
        routes = flat_probabilities.take(tree_arange + current_indices[None, ...])
        all_routes = [routes.clone()]
        for i in range(1, tree_depth + 1):
            current_indices = batch_indices + increment_masks[i]
            current_indices = tree_arange + current_indices[None, ...]
            current_routes = flat_probabilities.take(current_indices)
            all_routes.append(current_routes)
            routes *= current_routes
        ctx.save_for_backward(ones, sigmoid_net, *all_routes)
        ctx.tree_depth = tree_depth
        ctx.num_tree = num_tree
        return routes

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[Tensor], ...]:
        grad_output = grad_outputs[0]
        dummy_grads = tuple(None for _ in range(8))
        if grad_output is None:
            return (None,) + dummy_grads
        num_tree = ctx.num_tree
        tree_depth = ctx.tree_depth
        ones_list, sigmoid_net, *all_routes = ctx.saved_tensors
        cursor = 0
        divide = 1
        num_leaves = 2 ** (tree_depth + 1)
        sub_grads_shape = num_tree, sigmoid_net.shape[0], num_leaves - 1
        sub_grads = torch.zeros(*sub_grads_shape, device=grad_output.device)
        for i in range(tree_depth + 1):
            ones = ones_list[i]
            nodes = ones[None, None, ...]
            for j in range(tree_depth + 1):
                if j == i:
                    continue
                nodes = nodes * all_routes[j]
            sub_grad = grad_output * nodes
            section = int(round(num_leaves / divide))
            sub_grad = sub_grad.view(num_tree, -1, divide, section)
            sub_grad = sub_grad.sum(-1)
            sub_grads[..., cursor : cursor + divide] = sub_grad
            cursor += divide
            divide *= 2

        sub_grads = sub_grads.transpose(0, 1).contiguous()
        sub_grads = sub_grads.view(-1, num_tree * (num_leaves - 1))
        return (sigmoid_net * (1.0 - sigmoid_net) * sub_grads,) + dummy_grads


class DNDF(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int],
        *,
        num_tree: int = 10,
        tree_depth: int = 4,
        is_regression: Optional[bool] = None,
        tree_proj_config: Optional[Dict[str, Any]] = None,
        use_fast_dndf: bool = True,
    ):
        super().__init__()
        self._num_tree = num_tree
        self._tree_depth = tree_depth
        self._is_regression = is_regression
        if out_dim is not None and is_regression is None:
            self._is_regression = out_dim == 1
        self._num_leaf = 2 ** (self._tree_depth + 1)
        self._num_internals = self._num_leaf - 1
        self._output_dim = out_dim
        self._fast = use_fast_dndf
        if tree_proj_config is None:
            tree_proj_config = {}
        tree_proj_config.setdefault("pruner_config", {})
        self.tree_proj = Linear(
            in_dim,
            self._num_internals * self._num_tree,
            **tree_proj_config,
        )
        if out_dim is None:
            self.leaves = None
        else:
            leaves_shape = self._num_tree * self._num_leaf, out_dim
            self.leaves = nn.Parameter(torch.empty(*leaves_shape))
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(self.leaves.data)
        # buffers
        num_repeat, num_local_internals = self._num_leaf // 2, 1
        ones_np = np.repeat([1, -1], num_repeat)
        ones_list = [torch.from_numpy(ones_np.astype(np.float32))]
        increment_indices_np = np.repeat([0, self._num_internals], num_repeat)
        increment_indices = [torch.from_numpy(increment_indices_np.astype(np.int64))]
        for i in range(1, self._tree_depth + 1):
            num_repeat //= 2
            num_local_internals *= 2
            arange = np.arange(num_local_internals - 1, 2 * num_local_internals - 1)
            ones_np = np.repeat([1, -1], num_repeat)
            ones_np = np.tile(ones_np, 2**i)
            ones_list.append(torch.from_numpy(ones_np.astype(np.float32)))
            increment_mask = np.repeat(arange, 2)
            increment_mask += np.tile([0, self._num_internals], num_local_internals)
            increment_mask = np.repeat(increment_mask, num_repeat)
            increment_mask_ = torch.from_numpy(increment_mask.astype(np.int64))
            increment_indices.append(increment_mask_)
        self.increment_masks: Tensor
        self.register_buffer("tree_arange", torch.arange(num_tree)[..., None, None])
        self.register_buffer("ones", torch.stack(ones_list))
        self.register_buffer("increment_indices", torch.stack(increment_indices))

    def forward(self, net: Tensor) -> Tensor:
        num_batch = net.shape[0]
        tree_net = self.tree_proj(net)

        num_flat_prob = 2 * self._num_internals
        arange_args = 0, num_flat_prob * num_batch, num_flat_prob
        batch_indices = torch.arange(*arange_args, device=tree_net.device).view(-1, 1)

        if self._fast:
            routes = Route.apply(
                tree_net,
                self.tree_arange,
                batch_indices,
                self.ones,
                self.increment_indices,
                self._num_tree,
                num_batch,
                self._tree_depth,
                self._num_internals,
            )
        else:
            shape = num_batch, -1, self._num_internals
            p_left = torch.sigmoid(tree_net).view(*shape).transpose(0, 1)
            p_right = 1.0 - p_left
            flat_probabilities = torch.cat([p_left, p_right], dim=-1).contiguous()
            flat_probabilities = flat_probabilities.view(self._num_tree, -1)
            current_indices = batch_indices + self.increment_indices[0]  # type: ignore
            flat_dim = flat_probabilities.shape[-1]
            tree_arange = self.tree_arange * flat_dim  # type: ignore
            routes = flat_probabilities.take(tree_arange + current_indices[None, ...])
            for i in range(1, self._tree_depth + 1):
                current_indices = batch_indices + self.increment_indices[i]  # type: ignore
                current_indices = tree_arange + current_indices[None, ...]
                routes *= flat_probabilities.take(current_indices)

        features = routes.transpose(0, 1).contiguous().view(num_batch, -1)
        if self.leaves is None or self._output_dim is None:
            return features.view(num_batch, self._num_tree, -1)
        if self._is_regression or self._output_dim <= 1:
            outputs = features.mm(self.leaves)
        else:
            if self._fast:
                outputs = LeafAggregation.apply(features, self.leaves)
            else:
                leaves = F.softmax(self.leaves, dim=1)
                outputs = features.mm(leaves)
        return outputs / self._num_tree


class Pruner(Module):
    def __init__(self, config: Dict[str, Any], w_shape: Optional[List[int]] = None):
        super().__init__()
        self.eps: Tensor
        self.exp: Tensor
        self.alpha: Union[Tensor, nn.Parameter]
        self.beta: Union[Tensor, nn.Parameter]
        self.gamma: Union[Tensor, nn.Parameter]
        self.max_ratio: Union[Tensor, nn.Parameter]
        tensor = partial(torch.tensor, dtype=torch.float32)
        self.method = config.setdefault("method", "auto_prune")
        if self.method == "surgery":
            if w_shape is None:
                msg = "`w_shape` of `Pruner` should be provided when `surgery` is used"
                raise ValueError(msg)
            self.register_buffer("mask", torch.ones(*w_shape, dtype=torch.float32))
            self.register_buffer("alpha", tensor([config.setdefault("alpha", 1.0)]))
            self.register_buffer("beta", tensor([config.setdefault("beta", 4.0)]))
            self.register_buffer("gamma", tensor([config.setdefault("gamma", 1e-4)]))
            self.register_buffer("eps", tensor([config.setdefault("eps", 1e-12)]))
            keys = ["alpha", "beta", "gamma", "eps"]
        elif self.method == "simplified":
            self.register_buffer("alpha", tensor([config.setdefault("alpha", 0.01)]))
            self.register_buffer("beta", tensor([config.setdefault("beta", 1.0)]))
            self.register_buffer(
                "max_ratio", tensor([config.setdefault("max_ratio", 1.0)])
            )
            self.register_buffer("exp", tensor([config.setdefault("exp", 0.5)]))
            keys = ["alpha", "beta", "max_ratio", "exp"]
        else:
            self.register_buffer(
                "alpha",
                tensor(
                    [
                        config.setdefault(
                            "alpha", 1e-4 if self.method == "hard_prune" else 1e-2
                        )
                    ]
                ),
            )
            self.register_buffer("beta", tensor([config.setdefault("beta", 1.0)]))
            self.register_buffer("gamma", tensor([config.setdefault("gamma", 1.0)]))
            self.register_buffer(
                "max_ratio", tensor([config.setdefault("max_ratio", 1.0)])
            )
            if not all(
                scalar > 0
                for scalar in [self.alpha, self.beta, self.gamma, self.max_ratio]
            ):
                raise ValueError("parameters should greater than 0. in pruner")
            self.register_buffer("eps", tensor([config.setdefault("eps", 1e-12)]))
            if self.method == "auto_prune":
                for attr in ["alpha", "beta", "gamma", "max_ratio"]:
                    setattr(self, attr, torch.log(torch.exp(getattr(self, attr)) - 1))
                self.alpha, self.beta, self.gamma, self.max_ratio = map(
                    lambda param: nn.Parameter(param),
                    [self.alpha, self.beta, self.gamma, self.max_ratio],
                )
            keys = ["alpha", "beta", "gamma", "max_ratio", "eps"]
        self._repr_keys = keys

    def forward(self, w: Tensor) -> Tensor:
        w_abs = torch.abs(w)
        if self.method == "surgery":
            mu, std = torch.mean(w_abs), torch.std(w_abs)
            zeros_mask = self.mask == 0.0
            ones_mask = self.mask == 1.0
            to_zeros_mask = ones_mask & (w_abs <= 0.9 * (mu - self.beta * std))  # type: ignore
            to_ones_mask = zeros_mask & (w_abs >= 1.1 * (mu + self.beta * std))  # type: ignore
            self.mask.masked_fill(to_zeros_mask, 0.0)  # type: ignore
            self.mask.masked_fill(to_ones_mask, 1.0)  # type: ignore
            mask = self.mask
            del mu, std, ones_mask, zeros_mask, to_zeros_mask, to_ones_mask
        else:
            if self.method != "auto_prune":
                alpha, beta, ratio = self.alpha, self.beta, self.max_ratio
            else:
                alpha, beta, ratio = map(
                    F.softplus,
                    [self.alpha, self.beta, self.max_ratio],
                )
            if self.method == "simplified":
                log_w = torch.min(ratio, beta * torch.pow(w_abs, self.exp))
            else:
                w_abs_mean = torch.mean(w_abs)
                if self.method != "auto_prune":
                    gamma = self.gamma
                else:
                    gamma = F.softplus(self.gamma)
                log_w = torch.log(torch.max(self.eps, w_abs / (w_abs_mean * gamma)))
                log_w = torch.min(ratio, beta * log_w)
                del w_abs_mean
            mask = torch.max(alpha / beta * log_w, log_w)
            del log_w
        w = w * mask
        del w_abs, mask
        return w

    def extra_repr(self) -> str:
        if self.method == "auto_prune":
            return f"method='{self.method}'"
        max_str_len = max(map(len, self._repr_keys))
        return "\n".join(
            [f"(0): method={self.method}\n(1): Settings("]
            + [
                f"  {key:<{max_str_len}s} - {getattr(self, key).item()}"
                for key in self._repr_keys
            ]
            + [")"]
        )


class AttentionOutput(NamedTuple):
    output: Tensor
    weights: Tensor


attentions: Dict[str, Type["Attention"]] = {}


class Attention(Module, WithRegister["Attention"]):
    d = attentions

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        qk_scale: Optional[float] = None,
        kv_same: Optional[bool] = None,
        qkv_bias_same: bool = True,
        is_self_attention: bool = False,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        activation: Optional[str] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        out_linear_config: Optional[Dict[str, Any]] = None,
        reduction_ratio: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        if kv_same is None:
            if k_dim is not None and v_dim is not None and k_dim != v_dim:
                kv_same = False
            else:
                kv_same = True
        self.kv_same = kv_same
        has_reduction = reduction_ratio is not None and reduction_ratio > 1
        self.qkv_same = is_self_attention and not has_reduction
        if not is_self_attention:
            self.k_dim = k_dim or input_dim
            self.v_dim = v_dim or self.k_dim
        else:
            if k_dim is not None and k_dim != input_dim:
                raise ValueError("self attention is used but `k_dim` != `input_dim`")
            if v_dim is not None and v_dim != input_dim:
                raise ValueError("self attention is used but `v_dim` != `input_dim`")
            self.k_dim = self.v_dim = input_dim
        self.embed_dim = embed_dim or input_dim

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scaling = qk_scale or float(self.head_dim) ** 0.5
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("`embed_dim` must be divisible by `num_heads`")

        if self.qkv_same:
            self.kv_w = self.q_w = self.k_w = self.v_w = None
            self.in_w = nn.Parameter(torch.empty(3 * self.embed_dim, input_dim))
            nn.init.trunc_normal_(self.in_w, std=0.02)
        elif kv_same:
            self.in_w = self.k_w = self.v_w = None
            self.q_w = nn.Parameter(torch.empty(self.embed_dim, input_dim))
            self.kv_w = nn.Parameter(torch.empty(2 * self.embed_dim, input_dim))
            nn.init.trunc_normal_(self.q_w, std=0.02)
            nn.init.trunc_normal_(self.kv_w, std=0.02)
        else:
            self.in_w = None
            self.q_w = nn.Parameter(torch.empty(self.embed_dim, input_dim))
            self.k_w = nn.Parameter(torch.empty(self.embed_dim, self.k_dim))
            self.v_w = nn.Parameter(torch.empty(self.embed_dim, self.v_dim))
            nn.init.xavier_uniform_(self.q_w)
            nn.init.xavier_uniform_(self.k_w)
            nn.init.xavier_uniform_(self.v_w)
        if not bias:
            self.q_bias = self.k_bias = self.v_bias = None
            self.kv_bias = self.qkv_bias = None
        elif not qkv_bias_same:
            self.kv_bias = self.qkv_bias = None
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.k_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        elif self.qkv_same or not kv_same:
            self.q_bias = self.k_bias = self.v_bias = self.kv_bias = None
            self.qkv_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))
        else:
            self.k_bias = self.v_bias = self.qkv_bias = None
            self.q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.kv_bias = nn.Parameter(torch.zeros(2 * self.embed_dim))

        if out_linear_config is None:
            out_linear_config = {}
        self.out_linear = Linear(self.embed_dim, input_dim, **out_linear_config)

        self.dropout = dropout
        self.activation = Activation.make(activation, activation_config)

        if not has_reduction:
            self.reduction = None
        else:
            self.reduction = nn.Sequential(
                Conv2d(
                    self.embed_dim,
                    self.embed_dim,
                    kernel_size=reduction_ratio,  # type: ignore
                    stride=reduction_ratio,  # type: ignore
                    padding=0,
                ),
                Lambda(lambda t: t.flatten(2).transpose(1, 2)),
                nn.LayerNorm(self.embed_dim),
            )

    def _to_heads(self, tensor: Tensor, determinate: bool) -> Tensor:
        seq_len = tensor.shape[1]
        if determinate:
            seq_len = int(seq_len)
        tensor = tensor.view(-1, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def _get_weights(self, raw_weights: Tensor) -> Tensor:
        # in most cases the softmax version is good enough
        return F.softmax(raw_weights, dim=-1)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        return weights

    def _reduce(self, net: Tensor, hw: Optional[Tuple[int, int]] = None) -> Tensor:
        if self.reduction is None:
            return net
        if hw is None:
            msg = "`hw` should be provided when `reduction` is applied"
            raise ValueError(msg)
        net = net.transpose(1, 2).contiguous()
        net = net.view(-1, net.shape[1], *hw)
        net = self.reduction(net)
        return net

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        hw: Optional[Tuple[int, int]] = None,
        mask: Optional[Tensor] = None,
        determinate: bool = False,
    ) -> AttentionOutput:
        # `mask` represents slots which will be zeroed
        if self.qkv_same:
            qkv = F.linear(q, self.in_w, self.qkv_bias)
            if not determinate:
                q, k, v = qkv.chunk(3, dim=-1)
            else:
                qkv = qkv.view(-1, int(q.shape[1]), 3, self.embed_dim)
                q, k, v = map(partial(torch.squeeze, dim=2), qkv.split(1, dim=2))
        elif self.kv_same:
            # B, Nq, Din -> B, Nq, D
            q = F.linear(q, self.q_w, self.q_bias)
            # B, Nk, Dk -> B, Nk, D
            if self.reduction is not None:
                if hw is None:
                    msg = "`hw` should be provided when `reduction` is applied"
                    raise ValueError(msg)
                k = self._reduce(k, hw)
            k, v = F.linear(k, self.kv_w, self.kv_bias).chunk(2, dim=-1)
        else:
            if self.qkv_bias is not None:
                q_bias, k_bias, v_bias = self.qkv_bias.chunk(3)
            else:
                q_bias = self.q_bias
                k_bias = self.k_bias
                v_bias = self.v_bias
            # B, Nq, Din -> B, Nq, D
            q = F.linear(q, self.q_w, q_bias)
            # B, Nk, Dk -> B, Nk, D
            k = F.linear(self._reduce(k, hw), self.k_w, k_bias)
            # B, Nv, Dv -> B, Nv, D
            v = F.linear(self._reduce(v, hw), self.v_w, v_bias)
        q, k, v = map(self.activation, [q, k, v])
        # B, N*, D -> B * N_head, N*, D_head
        q, k, v = map(self._to_heads, [q, k, v], [determinate] * 3)
        if mask is not None:
            # B, Nq, Nk -> B, N_head, Nq, Nk
            mask = mask.repeat(self.num_heads, 1, 1)
            mask = mask.view(-1, self.num_heads, *mask.shape[1:])
        # B, N_head, Nq, Nk
        raw_weights = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            raw_weights.masked_fill_(mask, float("-inf"))
        # scale
        raw_weights = raw_weights / self.scaling
        # B, N_head, Nq, Nk -> B, N_head, Nq, Nk
        weights = self._get_weights(raw_weights)
        if 0.0 < self.dropout < 1.0:
            weights = F.dropout(weights, self.dropout, self.training)
        weights = self._weights_callback(weights)
        # B, N_head, Nq, D_head
        output = torch.matmul(weights, v)
        # B, N_head, Nq, D_head -> B, Nq, N_head, D_head
        output = output.transpose(1, 2).contiguous()
        # B, Nq, N_head, D_head -> B, Nq, D
        seq_len = output.shape[1]
        if determinate:
            seq_len = int(seq_len)
        output = output.view(-1, seq_len, self.embed_dim)
        # B, Nq, D -> B, Nq, Din
        output = self.activation(self.out_linear(output))
        return AttentionOutput(output, weights)


Attention.register("basic")(Attention)


@Attention.register("decayed")
class DecayedAttention(Attention):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        *,
        seq_len: int,
        dropout: float = 0.0,
        is_self_attention: bool = False,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        activation: Optional[str] = None,
        activation_config: Optional[Dict[str, Any]] = None,
        out_linear_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            input_dim,
            num_heads,
            dropout=dropout,
            is_self_attention=is_self_attention,
            k_dim=k_dim,
            v_dim=v_dim,
            embed_dim=embed_dim,
            activation=activation,
            activation_config=activation_config,
            out_linear_config=out_linear_config,
        )
        mask = np.zeros([seq_len, seq_len], dtype=np.float32)
        for i in range(1, seq_len):
            np.fill_diagonal(mask[i:], i**2)
        mask_ = torch.from_numpy(mask)
        decayed_mask = torch.empty(num_heads, seq_len, seq_len)
        for i in range(num_heads):
            decayed_mask[i] = torch.exp(-(0.1 ** (i + 3)) * mask_)
        self.register_buffer("decayed_mask", decayed_mask)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        last_shapes = weights.shape[1:]
        weights = weights.view(-1, self.num_heads, *last_shapes)
        weights = weights * self.decayed_mask
        weights = weights / (torch.sum(weights, dim=3).unsqueeze(3) + 1.0e-8)
        return weights.view(-1, *last_shapes)


class PixelNorm(Module):
    def forward(self, net: Tensor) -> Tensor:
        return F.normalize(net, dim=1)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5, momentum: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None

    def forward(self, net: Tensor) -> Tensor:
        return adain_with_params(net, self.bias, self.weight)

    def extra_repr(self) -> str:
        return str(self.dim)


class NormFactory:
    def __init__(self, norm_type: Optional[str]):
        self.norm_type = norm_type

    @property
    def use_bias(self) -> bool:
        return self.norm_type is None or not self.norm_type.startswith("batch")

    @property
    def norm_base(self) -> Callable:
        norm_type = self.norm_type
        norm_layer: Union[Type[Module], Any]
        if norm_type == "batch_norm":
            norm_layer = BN
        elif norm_type == "layer_norm":
            norm_layer = LN
        elif norm_type == "adain":
            norm_layer = AdaptiveInstanceNorm2d
        elif norm_type == "layer":
            norm_layer = nn.LayerNorm
        elif norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_type == "batch1d":
            norm_layer = nn.BatchNorm1d
        elif norm_type == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm_type == "spectral":
            norm_layer = torch.nn.utils.spectral_norm
        elif norm_type == "pixel":
            norm_layer = PixelNorm
        elif norm_type is None:

            def norm_layer(_: Any, *__: Any, **___: Any) -> nn.Identity:
                return nn.Identity()

        else:
            msg = f"normalization layer '{norm_type}' is not found"
            raise NotImplementedError(msg)
        return norm_layer

    @property
    def default_config(self) -> Dict[str, Any]:
        norm_type = self.norm_type
        config: Dict[str, Any] = {}
        if norm_type == "batch":
            config = {"affine": True, "track_running_stats": True}
        elif norm_type == "instance":
            config = {"affine": False, "track_running_stats": False}
        elif norm_type == "layer" or norm_type == "layer_norm":
            config = {"eps": 1.0e-6}
        return config

    def make(self, *args: Any, **kwargs: Any) -> Module:
        kwargs = update_dict(kwargs, self.default_config)
        return self.norm_base(*args, **kwargs)

    def inject_to(
        self,
        dim: int,
        norm_kwargs: Dict[str, Any],
        current_blocks: List[Module],
        *subsequent_blocks: Module,
    ) -> None:
        if self.norm_type != "spectral":
            new_block = self.make(dim, **norm_kwargs)
            current_blocks.append(new_block)
        else:
            last_block = current_blocks[-1]
            last_block = self.make(last_block, **norm_kwargs)
            current_blocks[-1] = last_block
        current_blocks.extend(subsequent_blocks)


class Residual(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        return net + self.module(net, **kwargs)


class PreNorm(Module):
    def __init__(
        self,
        *dims: int,
        module: Module,
        norm_type: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.norms = ModuleList([])
        for dim in dims:
            self.norms.append(NormFactory(norm_type).make(dim, **norm_kwargs))
        self.module = module

    def forward(self, *xs: Tensor, **kwargs: Any) -> Tensor:
        x_list = [norm(x) for x, norm in zip(xs, self.norms)]
        if not issubclass(self.module.__class__, Attention):
            return self.module(*x_list, **kwargs)
        if len(x_list) == 1:
            x_list = [x_list[0]] * 3
        elif len(x_list) == 2:
            x_list.append(x_list[1])
        if len(x_list) != 3:
            raise ValueError("there should be three inputs for `Attention`")
        return_attention = kwargs.pop("return_attention", False)
        attention_outputs = self.module(*x_list, **kwargs)
        if return_attention:
            return attention_outputs.weights
        return attention_outputs.output


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
        B = net.shape[0]
        in_latent = torch.repeat_interleave(self.in_latent[None, ...], B, dim=0)
        for cross_attn, cross_ff, self_attn_blocks in self.layers:
            in_latent = cross_attn(in_latent, net, mask=mask) + in_latent
            in_latent = cross_ff(in_latent) + in_latent
            for self_attn, self_ff in self_attn_blocks:
                in_latent = self_attn(in_latent) + in_latent
                in_latent = self_ff(in_latent) + in_latent
        if self.out_latent is not None:
            out_queries = torch.repeat_interleave(self.out_latent[None, ...], B, dim=0)
        if out_queries is None:
            return in_latent
        return self.decoder_cross_attn(out_queries, in_latent)


# mappings


mapping_dict: Dict[str, Type["MappingBase"]] = {}


class Linear(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        pruner_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias)
        if pruner_config is None:
            pruner = None
        else:
            pruner = Pruner(pruner_config, [out_dim, in_dim])
        self.config, self.pruner = shallow_copy_dict(kwargs), pruner
        with torch.no_grad():
            gain = 1.0 / math.sqrt(2.0)
            nn.init.xavier_normal_(self.linear.weight.data, gain)
            if bias:
                self.linear.bias.data.zero_()

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.linear.bias

    def forward(self, net: Tensor) -> Tensor:
        weight = self.linear.weight
        if self.pruner is not None:
            weight = self.pruner(weight)
        return F.linear(net, weight, self.linear.bias)


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
            self.activation: Optional[Module] = None
        else:
            activation_config = self.config.setdefault("activation_config", None)
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
        latent_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        init_method: str = "xavier_uniform",
        **kwargs: Any,
    ):
        super().__init__()
        # input mapping
        if in_dim == latent_dim:
            self.to_latent = nn.Identity()
        else:
            self.to_latent = Linear(
                in_dim,
                latent_dim,
                bias=True if bias is None else bias,
                pruner_config=pruner_config,
                init_method=init_method,
                **kwargs,
            )
        # residual unit
        self.residual_unit = nn.Sequential(
            BN(latent_dim),
            Activation.make(activation, kwargs.setdefault("activation_config", None)),
            nn.Identity() if not 0.0 < dropout < 1.0 else nn.Dropout(dropout),
            Mapping(
                latent_dim,
                latent_dim,
                bias=bias,
                pruner_config=pruner_config,
                dropout=dropout,
                batch_norm=batch_norm,
                activation=activation,
                init_method=init_method,
                **kwargs,
            ),
            Linear(
                latent_dim,
                latent_dim,
                bias=True if bias is None else bias,
                pruner_config=pruner_config,
                init_method=init_method,
                **kwargs,
            ),
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
        latent_dim: int,
        *,
        bias: Optional[bool] = None,
        pruner_config: Optional[dict] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        activation: Optional[str] = "ReLU",
        init_method: str = "xavier_uniform",
        **kwargs: Any,
    ):
        super().__init__()
        self.linear_mapping = Linear(
            in_dim,
            latent_dim,
            bias=True if bias is None else bias,
            pruner_config=pruner_config,
            init_method=init_method,
            **kwargs,
        )
        self.nonlinear_mapping = Mapping(
            in_dim,
            latent_dim,
            bias=bias,
            pruner_config=pruner_config,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            init_method=init_method,
            **kwargs,
        )
        self.gate_linear = Linear(
            in_dim,
            latent_dim,
            bias=True if bias is None else bias,
            pruner_config=pruner_config,
            init_method=init_method,
            **kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, net: Tensor) -> Tensor:
        linear = self.linear_mapping(net)
        nonlinear = self.nonlinear_mapping(net)
        gate = self.sigmoid(self.gate_linear(net))
        return gate * nonlinear + (1.0 - gate) * linear


# mixed stacks


to_patches: Dict[str, Type["ImgToPatches"]] = {}


class ImgToPatches(Module, WithDeviceMixin, WithRegister["ImgToPatches"]):
    d = to_patches

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        """should return patches and its hw"""

    @property
    def num_patches(self) -> int:
        shape = 1, self.in_channels, self.img_size, self.img_size
        with eval_context(self):
            net = self.forward(torch.zeros(*shape, device=self.device))[0]
        return net.shape[1]

    @staticmethod
    def _flatten(net: Tensor, determinate: bool) -> Tuple[Tensor, Any]:
        c, h, w = net.shape[1:]
        if determinate:
            c, h, w = map(int, [c, h, w])
        net = net.view(-1, c, h * w).transpose(1, 2).contiguous()
        return net, (h, w)


@ImgToPatches.register("vanilla")
class VanillaPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 128,
        **conv_kwargs: Any,
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        if img_size % patch_size != 0:
            raise ValueError(
                f"`img_size` ({img_size}) should be "
                f"divisible by `patch_size` ({patch_size})"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.projection = Conv2d(
            in_channels,
            latent_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            **conv_kwargs,
        )

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.projection(net)
        return self._flatten(net, determinate)


@ImgToPatches.register("overlap")
class OverlapPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_dim: int = 768,
        stride: int = 4,
        **conv_kwargs: Any,
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        self.conv = Conv2d(
            in_channels,
            latent_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size // 2, patch_size // 2),
            **conv_kwargs,
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.conv(net)
        net, hw = self._flatten(net, determinate)
        net = self.norm(net)
        return net, hw


@ImgToPatches.register("conv")
class ConvPatchEmbed(ImgToPatches):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        latent_channels: int = 64,
        latent_dim: int = 384,
        padding: Optional[int] = None,
        stride: Optional[int] = None,
        bias: bool = False,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__(img_size, patch_size, in_channels, latent_dim)
        latent_channels_list = [latent_channels] * (num_layers - 1)
        num_channels_list = [in_channels] + latent_channels_list + [latent_dim]
        if padding is None:
            padding = max(1, patch_size // 2)
        if stride is None:
            stride = max(1, (patch_size // 2) - 1)
        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    *get_conv_blocks(
                        num_channels_list[i],
                        num_channels_list[i + 1],
                        patch_size,
                        stride,
                        bias=bias,
                        activation=activation,
                        padding=padding,
                    ),
                    nn.MaxPool2d(3, 2, 1),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tuple[Tensor, Any]:
        net = self.conv(net)
        return self._flatten(net, determinate)


token_mixers: Dict[str, Type["TokenMixerBase"]] = {}


class TokenMixerBase(Module, WithRegister["TokenMixerBase"]):
    d = token_mixers

    def __init__(self, num_tokens: int, latent_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        pass


@TokenMixerBase.register("mlp")
class MLPTokenMixer(TokenMixerBase):
    def __init__(self, num_tokens: int, latent_dim: int, *, dropout: float = 0.1):
        super().__init__(num_tokens, latent_dim)
        self.net = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2), name="to_token_mixing"),
            FeedForward(num_tokens, num_tokens, dropout),
            Lambda(lambda x: x.transpose(1, 2), name="to_channel_mixing"),
        )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.net(net)


@TokenMixerBase.register("fourier")
class FourierTokenMixer(TokenMixerBase):
    def __init__(self, num_tokens: int, latent_dim: int):
        super().__init__(num_tokens, latent_dim)
        self.net = Lambda(lambda x: fft(fft(x, dim=-1), dim=-2).real, name="fourier")

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.net(net)


@TokenMixerBase.register("attention")
class AttentionTokenMixer(TokenMixerBase):
    def __init__(
        self,
        num_tokens: int,
        latent_dim: int,
        *,
        attention_type: str = "basic",
        **attention_kwargs: Any,
    ):
        super().__init__(num_tokens, latent_dim)
        attention_kwargs.setdefault("bias", False)
        attention_kwargs.setdefault("num_heads", 8)
        attention_kwargs["input_dim"] = latent_dim
        attention_kwargs.setdefault("is_self_attention", True)
        base = Attention.get(attention_type)
        self.net = base(**filter_kw(base, attention_kwargs))

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        kw = dict(hw=hw, mask=mask, determinate=determinate)
        return self.net(net, net, net, **kw).output


@TokenMixerBase.register("pool")
class PoolTokenMixer(TokenMixerBase):
    def __init__(self, num_tokens: int, latent_dim: int, *, pool_size: int = 3):
        super().__init__(num_tokens, latent_dim)
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
    ) -> Tensor:
        return self.pool(net) - net


ffn_dict: Dict[str, Type["FFN"]] = {}


class FFN(Module, WithRegister["FFN"]):
    d = ffn_dict

    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

    @property
    @abstractmethod
    def need_2d(self) -> bool:
        pass

    @abstractmethod
    def forward(self, net: Tensor) -> Tensor:
        pass


@FFN.register("ff")
class FeedForward(FFN):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        dropout: float,
        activation: str = "GELU",
    ):
        super().__init__(in_dim, latent_dim, dropout)
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            Activation.make(activation),
            nn.Dropout(dropout),
            Linear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    @property
    def need_2d(self) -> bool:
        return False

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


@FFN.register("mix_ff")
class MixFeedForward(FFN):
    def __init__(self, in_dim: int, latent_dim: int, dropout: float):
        super().__init__(in_dim, latent_dim, dropout)
        self.net = nn.Sequential(
            Linear(in_dim, latent_dim),
            Lambda(lambda t: t.permute(0, 3, 1, 2), "permute -> BCHW"),
            DepthWiseConv2d(latent_dim),
            Lambda(lambda t: t.flatten(2).transpose(1, 2), "transpose -> BNC"),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(latent_dim, in_dim),
            nn.Dropout(dropout),
        )

    @property
    def need_2d(self) -> bool:
        return True

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class DropPath(Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, net: Tensor) -> Tensor:
        if not 0.0 < self.dropout < 1.0 or not self.training:
            return net
        keep_prob = 1.0 - self.dropout
        shape = (net.shape[0],) + (1,) * (net.ndim - 1)
        rand = torch.rand(shape, dtype=net.dtype, device=net.device)
        random_tensor = keep_prob + rand
        random_tensor.floor_()
        net = net.div(keep_prob) * random_tensor
        return net

    def extra_repr(self) -> str:
        return str(self.dropout)


class MixingBlock(Module):
    def __init__(
        self,
        num_tokens: int,
        latent_dim: int,
        feedforward_dim: int,
        *,
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_type: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        first_norm: Optional[nn.Module] = None,
        residual_after_norm: bool = False,
    ):
        super().__init__()
        self.first_norm = first_norm
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if token_mixing_config is None:
            token_mixing_config = {}
        token_mixing_config.update({"num_tokens": num_tokens, "latent_dim": latent_dim})
        self.token_mixing = PreNorm(
            latent_dim,
            module=TokenMixerBase.make(token_mixing_type, token_mixing_config),
            norm_type=norm_type,
            norm_kwargs=norm_kwargs,
        )
        if channel_mixing_config is None:
            channel_mixing_config = {}
        channel_mixing_config.update(
            {
                "in_dim": latent_dim,
                "latent_dim": feedforward_dim,
                "dropout": dropout,
            }
        )
        ffn = FFN.make(channel_mixing_type, channel_mixing_config)
        if residual_after_norm:
            factory = NormFactory(norm_type)
            self.channel_norm = factory.make(latent_dim, **(norm_kwargs or {}))
            self.channel_mixing = ffn
        else:
            self.channel_norm = None
            self.channel_mixing = PreNorm(
                latent_dim,
                module=ffn,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
            )

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        *,
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if self.first_norm is not None:
            net = self.first_norm(net)
        token_mixing_kw = dict(hw=hw, determinate=determinate)
        token_mixing_kw.update(kwargs)
        net = net + self.drop_path(self.token_mixing(net, **token_mixing_kw))
        if self.channel_norm is None:
            need_2d = self.channel_mixing.module.need_2d
        else:
            net = self.channel_norm(net)
            need_2d = self.channel_mixing.need_2d
        if not need_2d:
            channel_mixing_net = net
        else:
            if hw is None:
                raise ValueError("`hw` should be provided when FFN needs 2d input")
            channel_mixing_net = net.view(-1, *hw, net.shape[-1])
        net = net + self.drop_path(self.channel_mixing(channel_mixing_net))
        return net


class PositionalEncoding(Module):
    def __init__(
        self,
        dim: int,
        num_history: int,
        dropout: float = 0.0,
        *,
        num_heads: int,
        enable: bool = True,
    ):
        super().__init__()
        self.pos_drop = None
        self.pos_encoding = None
        if enable:
            self.pos_drop = nn.Dropout(p=dropout)
            self.pos_encoding = nn.Parameter(torch.zeros(1, num_history, dim))
            nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self.num_heads = num_heads

    def forward(
        self,
        net: Tensor,
        *,
        hwp: Optional[Tuple[int, int, int]] = None,
    ) -> Tensor:
        if self.pos_encoding is None or self.pos_drop is None:
            return net
        pos_encoding = self.interpolate_pos_encoding(net, hwp)
        pos_encoding = self.pos_drop(pos_encoding)
        return net + pos_encoding

    # this is for vision positional encodings
    def interpolate_pos_encoding(
        self,
        net: Tensor,
        hwp: Optional[Tuple[int, int, int]],
    ) -> Tensor:
        pos_encoding = self.pos_encoding
        assert pos_encoding is not None
        num_current_history = net.shape[1] - self.num_heads
        num_history = pos_encoding.shape[1] - self.num_heads
        if hwp is None:
            w = h = patch_size = None
        else:
            h, w, patch_size = hwp
        if num_current_history == num_history and w == h:
            return pos_encoding
        if w is None or h is None or patch_size is None:
            raise ValueError("`hwp` should be provided for `interpolate_pos_encoding`")
        head_encoding = None
        if self.num_heads > 0:
            head_encoding = pos_encoding[:, : self.num_heads]
            pos_encoding = pos_encoding[:, self.num_heads :]
        dim = net.shape[-1]
        # This assume that the original input is squared image
        sqrt = math.sqrt(num_history)
        wh_ratio = w / h
        pw = math.sqrt(num_current_history * wh_ratio) + 0.1
        ph = math.sqrt(num_current_history / wh_ratio) + 0.1
        pos_encoding = interpolate(
            pos_encoding.reshape(1, int(sqrt), int(sqrt), dim).permute(0, 3, 1, 2),
            factor=(pw / sqrt, ph / sqrt),
            mode="bicubic",
        )
        assert int(pw) == pos_encoding.shape[-2] and int(ph) == pos_encoding.shape[-1]
        pos_encoding = pos_encoding.permute(0, 2, 3, 1).view(1, -1, dim)
        if head_encoding is None:
            return pos_encoding
        return torch.cat([head_encoding, pos_encoding], dim=1)


class SequencePooling(Module):
    def __init__(self, dim: int, aux_heads: Optional[List[str]], bias: bool = True):
        super().__init__()
        self.out_dim = 1 + (0 if aux_heads is None else len(aux_heads))
        self.projection = Linear(dim, self.out_dim, bias=bias)

    def forward(self, net: Tensor) -> Tensor:
        weights = self.projection(net)
        weights = F.softmax(weights, dim=1).transpose(-1, -2)
        net = torch.matmul(weights, net)
        if self.out_dim > 1:
            return net
        return net.squeeze(-2)


class MixedStackedEncoder(Module):
    def __init__(
        self,
        dim: int,
        num_history: int,
        *,
        token_mixing_type: str,
        token_mixing_config: Optional[Dict[str, Any]] = None,
        channel_mixing_type: str = "ff",
        channel_mixing_config: Optional[Dict[str, Any]] = None,
        num_layers: int = 4,
        dropout: float = 0.0,
        dpr_list: Optional[List[float]] = None,
        drop_path_rate: float = 0.1,
        norm_type: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        first_norm: Optional[nn.Module] = None,
        residual_after_norm: bool = False,
        feedforward_dim_ratio: float = 1.0,
        reduce_head: bool = True,
        sequence_pool: bool = False,
        use_head_token: bool = False,
        use_positional_encoding: bool = False,
        norm_after_head: bool = False,
        aux_heads: Optional[List[str]] = None,
    ):
        super().__init__()
        # head token
        self.aux_heads = aux_heads
        if not use_head_token:
            num_heads = 0
            self.head_token = None
        else:
            num_heads = 1
            if aux_heads is not None:
                num_heads += len(aux_heads)
            self.head_token = nn.Parameter(torch.zeros(1, num_heads, dim))
        self.num_heads = num_heads
        # positional encoding
        num_history += num_heads
        self.pos_encoding = PositionalEncoding(
            dim,
            num_history,
            dropout,
            num_heads=num_heads,
            enable=use_positional_encoding,
        )
        # core
        if dpr_list is None:
            dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        feedforward_dim = int(round(dim * feedforward_dim_ratio))
        self.mixing_blocks = ModuleList(
            [
                MixingBlock(
                    num_history,
                    dim,
                    feedforward_dim,
                    token_mixing_type=token_mixing_type,
                    token_mixing_config=token_mixing_config,
                    channel_mixing_type=channel_mixing_type,
                    channel_mixing_config=channel_mixing_config,
                    dropout=dropout,
                    drop_path=drop_path,
                    norm_type=norm_type,
                    norm_kwargs=norm_kwargs,
                    first_norm=first_norm if i == 0 else None,
                    residual_after_norm=residual_after_norm,
                )
                for i, drop_path in enumerate(dpr_list)
            ]
        )
        # head
        if self.head_token is not None:
            if self.aux_heads is None:
                head = Lambda(lambda x: x[:, 0], name="head_token")
            else:
                head = Lambda(lambda x: x[:, : self.num_heads], name="head_token")
        elif sequence_pool:
            head = SequencePooling(dim, aux_heads)
        else:
            if aux_heads is not None:
                raise ValueError(
                    "either `head_token` or `sequence_pool` should be used "
                    f"when `aux_heads` ({aux_heads}) is provided"
                )
            if not reduce_head:
                head = nn.Identity()
            else:
                head = Lambda(lambda x: x.mean(1), name="global_average")
        if norm_after_head:
            self.head_norm = NormFactory(norm_type).make(dim, **(norm_kwargs or {}))
            self.head = head
        else:
            self.head_norm = None
            self.head = PreNorm(
                dim,
                module=head,
                norm_type=norm_type,
                norm_kwargs=norm_kwargs,
            )
        # initializations
        if self.head_token is not None:
            nn.init.trunc_normal_(self.head_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        if isinstance(m, nn.Linear) or isinstance(m, Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def pre_process(
        self,
        net: Tensor,
        *,
        determinate: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        n, t, d = net.shape
        if self.head_token is not None:
            head_tokens = self.head_token.repeat([n, 1, 1])
            net = torch.cat([head_tokens, net], dim=1)
            t += 1
        if determinate:
            net = net.view(-1, *map(int, [t, d]))
        net = self.pos_encoding(net, **kwargs)
        return net

    def post_process(self, net: Tensor) -> Tensor:
        net = self.head(net)
        if self.head_norm is not None:
            net = self.head_norm(net)
        return net

    def forward(
        self,
        net: Tensor,
        hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> Tensor:
        determinate = kwargs.pop("determinate", False)
        net = self.pre_process(net, determinate=determinate, **kwargs)
        for block in self.mixing_blocks:
            net = block(net, hw, determinate=determinate)
        net = self.post_process(net)
        return net


# cv


class GaussianBlur3(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        base = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = base[:, None] * base[None, :] / 16.0
        kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.kernel: Tensor
        self.register_buffer("kernel", kernel)
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernel, groups=self.in_channels, padding=1)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: Any = "same",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        weight_scale: Optional[float] = None,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        self.kernel_size = kernel_size
        self.reflection_pad = None
        if padding == "same":
            padding = kernel_size // 2
        elif isinstance(padding, str) and padding.startswith("reflection"):
            reflection_padding: Any
            if padding == "reflection":
                reflection_padding = kernel_size // 2
            else:
                reflection_padding = int(padding[len("reflection") :])
            padding = 0
            if transform_kernel:
                reflection_padding = [reflection_padding] * 4
                reflection_padding[0] += 1
                reflection_padding[2] += 1
            self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.groups, self.stride = groups, stride
        self.dilation, self.padding = dilation, padding
        self.transform_kernel = transform_kernel
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kernel_size,
                kernel_size,
            )
        )
        if not bias:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.demodulate = demodulate
        self.weight_scale = weight_scale
        # initialize
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight.data, gain / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.zero_()

    def _same_padding(self, size: int) -> int:
        stride = self.stride
        dilation = self.dilation
        return ((size - 1) * (stride - 1) + dilation * (self.kernel_size - 1)) // 2

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
    ) -> Tensor:
        b, c, *hw = net.shape
        # padding
        padding = self.padding
        if self.padding == "same":
            padding = tuple(map(self._same_padding, hw))
        if self.reflection_pad is not None:
            net = self.reflection_pad(net)
        # transform kernel
        w: Union[nn.Parameter, Tensor] = self.weight
        if self.transform_kernel:
            w = F.pad(w, [1, 1, 1, 1], mode="constant")
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            ) * 0.25
        # ordinary convolution
        if style is None:
            bias = self.bias
            groups = self.groups
        # 'stylized' convolution, used in StyleGAN
        else:
            suffix = "when `style` is provided"
            if self.bias is not None:
                raise ValueError(f"`bias` should not be used {suffix}")
            if self.groups != 1:
                raise ValueError(f"`groups` should be 1 {suffix}")
            if self.reflection_pad is not None:
                raise ValueError(
                    f"`reflection_pad` should not be used {suffix}, "
                    "maybe you want to use `same` padding?"
                )
            w = w[None, ...] * style[..., None, :, None, None]
            # prepare for group convolution
            bias = None
            groups = b
            net = net.view(1, -1, *hw)  # 1, b*in, h, w
            w = w.view(b * self.out_c, *w.shape[2:])  # b*out, in, wh, ww
        if self.demodulate:
            w = w * torch.rsqrt(w.pow(2).sum([-3, -2, -1], keepdim=True) + 1e-8)
        if self.weight_scale is not None:
            w = w * self.weight_scale
        # conv core
        if not transpose:
            fn = F.conv2d
        else:
            fn = F.conv_transpose2d
            if groups == 1:
                w = w.transpose(0, 1)
            else:
                oc, ic, kh, kw = w.shape
                w = w.reshape(groups, oc // groups, ic, kh, kw)
                w = w.transpose(1, 2)
                w = w.reshape(groups * ic, oc // groups, kh, kw)
        net = fn(
            net,
            w,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
        )
        if style is None:
            return net
        return net.view(b, -1, *net.shape[2:])

    def extra_repr(self) -> str:
        return (
            f"{self.in_c}, {self.out_c}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"bias={self.bias is not None}, demodulate={self.demodulate}"
        )


class DepthWiseConv2d(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        super().__init__()
        if isinstance(output_size, int):
            output_size = output_size, output_size
        self.h, self.w = output_size

    def forward(self, net: Tensor) -> Tensor:
        h, w = map(int, net.shape[2:])
        sh, sw = map(math.floor, [h / self.h, w / self.w])
        kh = h - (self.h - 1) * sh
        kw = w - (self.w - 1) * sw
        return F.avg_pool2d(net, kernel_size=(kh, kw), stride=(sh, sw))


class MaxUnpool2d_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return F.max_unpool2d(*args)

    @staticmethod
    def symbolic(
        g: Any,
        net: Any,
        indices: Any,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        output_size: Any,
    ) -> Any:
        # get shape
        input_shape = g.op("Shape", net)
        const_0 = g.op("Constant", value_t=torch.tensor(0))
        const_1 = g.op("Constant", value_t=torch.tensor(1))
        batch_size = g.op("Gather", input_shape, const_0, axis_i=0)
        channel = g.op("Gather", input_shape, const_1, axis_i=0)

        # height = (height - 1) * stride + kernel_size
        height = g.op(
            "Gather",
            input_shape,
            g.op("Constant", value_t=torch.tensor(2)),
            axis_i=0,
        )
        height = g.op("Sub", height, const_1)
        height = g.op("Mul", height, g.op("Constant", value_t=torch.tensor(stride[1])))
        height = g.op(
            "Add",
            height,
            g.op("Constant", value_t=torch.tensor(kernel_size[1])),
        )

        # width = (width - 1) * stride + kernel_size
        width = g.op(
            "Gather",
            input_shape,
            g.op("Constant", value_t=torch.tensor(3)),
            axis_i=0,
        )
        width = g.op("Sub", width, const_1)
        width = g.op("Mul", width, g.op("Constant", value_t=torch.tensor(stride[0])))
        width = g.op(
            "Add",
            width,
            g.op("Constant", value_t=torch.tensor(kernel_size[0])),
        )

        # step of channel
        channel_step = g.op("Mul", height, width)
        # step of batch
        batch_step = g.op("Mul", channel_step, channel)

        # channel offset
        range_channel = g.op("Range", const_0, channel, const_1)
        range_channel = g.op(
            "Reshape",
            range_channel,
            g.op("Constant", value_t=torch.tensor([1, -1, 1, 1])),
        )
        range_channel = g.op("Mul", range_channel, channel_step)
        range_channel = g.op("Cast", range_channel, to_i=7)  # 7 is int64

        # batch offset
        range_batch = g.op("Range", const_0, batch_size, const_1)
        range_batch = g.op(
            "Reshape",
            range_batch,
            g.op("Constant", value_t=torch.tensor([-1, 1, 1, 1])),
        )
        range_batch = g.op("Mul", range_batch, batch_step)
        range_batch = g.op("Cast", range_batch, to_i=7)  # 7 is int64

        # update indices
        indices = g.op("Add", indices, range_channel)
        indices = g.op("Add", indices, range_batch)

        return g.op(
            "MaxUnpool",
            net,
            indices,
            kernel_shape_i=kernel_size,
            strides_i=stride,
        )


class MaxUnpool2d(_MaxUnpoolNd):
    def __init__(self, kernel_size: Any, stride: Any = None, padding: Any = 0):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride or kernel_size)
        self.padding = _pair(padding)

    def forward(self, net: Tensor, indices: Tensor, output_size: Any = None) -> Tensor:
        return MaxUnpool2d_op.apply(
            net,
            indices,
            self.kernel_size,
            self.stride,
            self.padding,
            output_size,
        )


class Interpolate(Module):
    def __init__(self, factor: Optional[float] = None, mode: str = "nearest"):
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.kw = dict(mode=mode, factor=factor)

    def forward(self, net: Tensor, *, determinate: bool = False) -> Tensor:
        if self.factor is not None:
            net = interpolate(net, determinate=determinate, **self.kw)  # type: ignore
        return net

    def extra_repr(self) -> str:
        return f"{self.factor}, {self.mode}"


class UpsampleConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        dilation: int = 1,
        mode: str = "nearest",
        padding: Optional[Union[int, str]] = None,
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        factor: Optional[float] = None,
    ):
        if mode == "transpose":
            if factor == 1.0:
                mode = "nearest"
                factor = None
            elif factor is not None:
                stride = int(round(factor))
                if padding is None:
                    padding = 0
        if padding is None:
            padding = "same"
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
            transform_kernel=transform_kernel,
            bias=bias,
            demodulate=demodulate,
        )
        if mode == "transpose":
            self.upsample = None
        else:
            self.upsample = Interpolate(factor, mode)

    def forward(
        self,
        net: Tensor,
        style: Optional[Tensor] = None,
        *,
        transpose: bool = False,
        determinate: bool = False,
    ) -> Tensor:
        if self.upsample is None:
            transpose = True
        else:
            net = self.upsample(net, determinate=determinate)
            if transpose:
                raise ValueError("should not use transpose when `upsample` is used")
        return super().forward(net, style, transpose=transpose)


class CABlock(Module):
    """Coordinate Attention"""

    def __init__(self, num_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        latent_channels = max(8, num_channels // reduction)
        self.conv_blocks = nn.Sequential(
            *get_conv_blocks(
                num_channels,
                latent_channels,
                kernel_size=1,
                stride=1,
                norm_type="batch",
                activation=Activation.make("h_swish"),
                padding=0,
            )
        )

        conv2d = lambda: Conv2d(
            latent_channels,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_h = conv2d()
        self.conv_w = conv2d()

    def forward(self, net: Tensor) -> Tensor:
        original = net

        n, c, h, w = net.shape
        net_h = self.pool_h(net)
        net_w = self.pool_w(net).transpose(2, 3)

        net = torch.cat([net_h, net_w], dim=2)
        net = self.conv_blocks(net)

        net_h, net_w = torch.split(net, [h, w], dim=2)
        net_w = net_w.transpose(2, 3)

        net_h = self.conv_h(net_h).sigmoid()
        net_w = self.conv_w(net_w).sigmoid()

        return original * net_w * net_h


class ECABlock(Module):
    """Efficient Channel Attention"""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, net: Tensor) -> Tensor:
        w = self.avg_pool(net).squeeze(-1).transpose(-1, -2)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1)
        w = self.sigmoid(w)
        return w * net


class SEBlock(Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        *,
        impl: str = "conv",
        block_impl: str = "cflearn",
    ):
        super().__init__()
        self.in_channels = in_channels
        if block_impl == "cflearn":
            conv_base = Conv2d
            self.avg_pool = AdaptiveAvgPool2d(1)
        elif block_impl == "torch":
            conv_base = nn.Conv2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"unrecognized `block_impl` ({block_impl}) found")
        self.fc = self.up = self.down = None
        if impl == "conv":
            self.down = conv_base(
                in_channels,
                latent_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            )
            self.up = conv_base(
                latent_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        elif impl == "fc":
            self.fc = nn.Sequential(
                nn.Linear(in_channels, latent_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(latent_channels, in_channels, bias=False),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"implementation '{impl}' is not recognized")

    def forward(self, net: Tensor) -> Tensor:
        inp = net
        net = self.avg_pool(net)
        if self.fc is not None:
            net = self.fc(net.view(-1, self.in_channels))
        elif self.up is not None and self.down is not None:
            net = self.down(net)
            net = F.relu(net)
            net = self.up(net)
            net = torch.sigmoid(net)
        net = net.view(-1, self.in_channels, 1, 1)
        return inp * net


def get_conv_blocks(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    *,
    bias: bool = True,
    demodulate: bool = False,
    norm_type: Optional[str] = None,
    norm_kwargs: Optional[Dict[str, Any]] = None,
    ca_reduction: Optional[int] = None,
    eca_kernel_size: Optional[int] = None,
    activation: Optional[Union[str, Module]] = None,
    conv_base: Type["Conv2d"] = Conv2d,
    pre_activate: bool = False,
    **conv2d_kwargs: Any,
) -> List[Module]:
    conv = conv_base(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias,
        demodulate=demodulate,
        **conv2d_kwargs,
    )
    blocks: List[Module] = []
    if not pre_activate:
        blocks.append(conv)
    if not demodulate:
        factory = NormFactory(norm_type)
        factory.inject_to(out_channels, norm_kwargs or {}, blocks)
    if eca_kernel_size is not None:
        blocks.append(ECABlock(kernel_size))
    if activation is not None:
        if isinstance(activation, str):
            activation = Activation.make(activation)
        blocks.append(activation)
    if ca_reduction is not None:
        blocks.append(CABlock(out_channels, ca_reduction))
    if pre_activate:
        blocks.append(conv)
    return blocks


class ResidualBlock(Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        ca_reduction: Optional[int] = None,
        eca_kernel_size: Optional[int] = None,
        norm_type: Optional[str] = "batch",
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["norm_type"] = norm_type
        k1 = shallow_copy_dict(kwargs)
        k1["ca_reduction"] = ca_reduction
        k1.setdefault("activation", nn.LeakyReLU(0.2, inplace=True))
        blocks = get_conv_blocks(dim, dim, kernel_size, stride, **k1)
        if 0.0 < dropout < 1.0:
            blocks.append(nn.Dropout(dropout))
        k2 = shallow_copy_dict(kwargs)
        k2["activation"] = None
        k2["eca_kernel_size"] = eca_kernel_size
        blocks.extend(get_conv_blocks(dim, dim, kernel_size, stride, **k2))
        self.net = Residual(nn.Sequential(*blocks))

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class ResidualBlockV2(Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        ca_reduction: Optional[int] = None,
        eca_kernel_size: Optional[int] = None,
        norm_type: Optional[str] = "batch",
        **kwargs: Any,
    ):
        super().__init__()
        kwargs["norm_type"] = norm_type
        k1 = shallow_copy_dict(kwargs)
        k1["pre_activate"] = True
        k1["ca_reduction"] = ca_reduction
        k1.setdefault("activation", nn.LeakyReLU(0.2, inplace=True))
        blocks = get_conv_blocks(dim, dim, kernel_size, stride, **k1)
        if 0.0 < dropout < 1.0:
            blocks.append(nn.Dropout(dropout))
        k2 = shallow_copy_dict(kwargs)
        k2["pre_activate"] = True
        k2["eca_kernel_size"] = eca_kernel_size
        blocks.extend(get_conv_blocks(dim, dim, kernel_size, stride, **k2))
        self.net = Residual(nn.Sequential(*blocks))

    def forward(self, net: Tensor) -> Tensor:
        return self.net(net)


class ChannelPadding(Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        map_dim: Optional[int] = None,
        *,
        is_1d: bool = False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.map_dim = map_dim
        self.is_global = map_dim is None
        self.is_conditional = num_classes is not None
        if self.is_global:
            map_dim = 1
        token_shape = (num_classes or 1), latent_channels, map_dim, map_dim
        self.channel_padding = nn.Parameter(torch.randn(*token_shape))  # type: ignore
        in_nc = in_channels + latent_channels
        out_nc = in_channels
        if is_1d:
            self.mapping = Linear(in_nc, out_nc, bias=False)
        else:
            self.mapping = Conv2d(in_nc, out_nc, kernel_size=1, bias=False)

    def forward(self, net: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if not self.is_conditional:
            padding = self.channel_padding.repeat(net.shape[0], 1, 1, 1)
        else:
            if labels is None:
                msg = "`labels` should be provided in conditional `ChannelPadding`"
                raise ValueError(msg)
            padding = self.channel_padding[labels.view(-1)]
        if self.is_global:
            if len(net.shape) == 2:
                padding = squeeze(padding)
            else:
                padding = padding.repeat(1, 1, *net.shape[-2:])
        net = torch.cat([net, padding], dim=1)
        net = self.mapping(net)
        return net

    def extra_repr(self) -> str:
        dim_str = f"{self.in_channels}+{self.latent_channels}"
        map_dim_str = "global" if self.is_global else f"{self.map_dim}x{self.map_dim}"
        return f"{dim_str}, {map_dim_str}"
