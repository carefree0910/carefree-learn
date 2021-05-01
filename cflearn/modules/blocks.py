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
from functools import partial
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict

from ..types import tensor_dict_type
from ..protocol import WithRegister
from ..constants import WARNING_PREFIX
from ..misc.toolkit import Initializer


# auxiliary


def _get_clones(module: Module, n: int) -> ModuleList:
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


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
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        net = super().forward(net)
        if len(net.shape) == 3:
            net = net.transpose(1, 2)
        return net


class Dropout(Module):
    def __init__(self, dropout: float):
        if dropout < 0.0 or dropout >= 1.0:
            msg = f"dropout probability has to be between [0, 1), but got {dropout}"
            raise ValueError(msg)
        super().__init__()
        self._keep_prob = 1.0 - dropout
        self._mask_cache: Optional[torch.Tensor] = None

    def forward(self, net: torch.Tensor, *, reuse: bool = False) -> torch.Tensor:
        if not self.training or self._keep_prob >= 1.0 - 1.0e-8:
            return net
        if reuse:
            mask = self._mask_cache
        else:
            self._mask_cache = mask = (
                torch.bernoulli(net.new(*net.shape).fill_(self._keep_prob))
                / self._keep_prob
            )
        net = net * mask
        del mask
        return net

    def extra_repr(self) -> str:
        return f"keep={self._keep_prob}"


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
                f"  {name:<{max_str_len}s} - torch.Tensor({list(param.shape)})"
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
    ) -> torch.Tensor:
        if not self.registered:
            raise ValueError("losses need to be registered")
        if naive or self._method is None:
            return self._naive(loss_dict)
        return getattr(self, f"_{self._method}")(loss_dict)

    @staticmethod
    def _naive(loss_dict: tensor_dict_type) -> torch.Tensor:
        return sum(loss_dict.values())  # type: ignore

    def _softmax(self, loss_dict: tensor_dict_type) -> torch.Tensor:
        assert self._slice is not None
        w = self.w if self._slice == self._n_task else self.w[: self._slice]
        softmax_w = nn.functional.softmax(w, dim=0)
        losses = []
        for key, loss in loss_dict.items():
            idx = self._registered.get(key)
            losses.append(loss if idx is None else loss * softmax_w[idx])
        final_loss: torch.Tensor = sum(losses)  # type: ignore
        return final_loss * self._slice

    def extra_repr(self) -> str:
        method = "naive" if self._method is None else self._method
        return f"n_task={self._n_task}, method='{method}'"


class _multiplied_activation(Module, metaclass=ABCMeta):
    def __init__(
        self,
        ratio: float,
        trainable: bool = True,
    ):
        super().__init__()
        self.trainable = trainable
        ratio_ = torch.tensor([ratio], dtype=torch.float32)
        self.ratio = ratio_ if not trainable else nn.Parameter(ratio_)

    @abstractmethod
    def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x * self.ratio)

    def extra_repr(self) -> str:
        return f"ratio={self.ratio.item()}, trainable={self.trainable}"


class Activations:
    """
    Wrapper class for pytorch activations
    * when pytorch implemented corresponding activation, it will be returned
    * otherwise, custom implementation will be returned

    Parameters
    ----------
    configs : {None, dict}, configuration for the activation

    Examples
    --------
    >>> act = Activations()
    >>> print(type(act.ReLU))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.module("ReLU")))  # <class 'nn.modules.activation.ReLU'>
    >>> print(type(act.Tanh))  # <class 'nn.modules.activation.Tanh'>
    >>> print(type(act.one_hot))  # <class '__main__.Activations.one_hot.<locals>.OneHot'>

    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        if configs is None:
            configs = {}
        self.configs = configs

    def __getattr__(self, item: str) -> Module:
        kwargs = self.configs.setdefault(item, {})
        try:
            return getattr(nn, item)(**kwargs)
        except AttributeError:
            func = getattr(torch, item, getattr(F, item, None))
            if func is None:
                raise NotImplementedError(
                    "neither pytorch nor custom Activations "
                    f"implemented activation '{item}'"
                )
            return Lambda(partial(func, **kwargs), item)

    def module(self, name: Optional[str]) -> Module:
        if name is None:
            return nn.Identity()
        return getattr(self, name)

    # publications

    @property
    def glu(self) -> Module:
        config = self.configs.setdefault("glu", {})
        in_dim = config.get("in_dim")
        if in_dim is None:
            raise ValueError("`in_dim` should be provided in glu")
        bias = config.setdefault("bias", True)

        class GLU(Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(in_dim, 2 * in_dim, bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                projection, gate = self.linear(x).chunk(2, dim=1)
                return projection * torch.sigmoid(gate)

        return GLU()

    @property
    def mish(self) -> Module:
        class Mish(Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * (torch.tanh(F.softplus(x)))

        return Mish()

    # custom

    # TODO : After updated to pytorch>=1.7.0, re-implement this
    @property
    def logit(self) -> Module:
        kwargs = self.configs.setdefault("logit", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _logit(x: torch.Tensor) -> torch.Tensor:
            x = torch.clamp(x, eps, 1.0 - eps)
            return torch.log(x / (1.0 - x))

        return Lambda(_logit, f"logit_{eps:.2e}")

    @property
    def atanh(self) -> Module:
        kwargs = self.configs.setdefault("atanh", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _atanh(x: torch.Tensor) -> torch.Tensor:
            return torch.atanh(torch.clamp(x, -1.0 + eps, 1.0 - eps))

        return Lambda(_atanh, f"atanh_{eps:.2e}")

    @property
    def isoftplus(self) -> Module:
        kwargs = self.configs.setdefault("isoftplus", {})
        eps = kwargs.setdefault("eps", 1.0e-6)

        def _isoftplus(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x.clamp_min(eps).exp() - 1.0)

        return Lambda(_isoftplus, f"isoftplus_{eps:.2e}")

    @property
    def sign(self) -> Module:
        config = self.configs.setdefault("sign", {})
        randomize_at_zero = config.setdefault("randomize_at_zero", False)
        eps = config.setdefault("eps", 1e-12)
        suffix = "_randomized" if randomize_at_zero else ""

        def _core(x: torch.Tensor) -> torch.Tensor:
            if randomize_at_zero:
                x = x + (2 * torch.empty_like(x).uniform_() - 1.0) * eps
            return torch.sign(x)

        return Lambda(_core, f"sign{suffix}")

    @property
    def one_hot(self) -> Module:
        f = lambda x: x * (x == torch.max(x, dim=1, keepdim=True)[0]).to(torch.float32)
        return Lambda(f, "one_hot")

    @property
    def sine(self) -> Module:
        return Lambda(lambda x: torch.sin(x), "sine")

    @property
    def multiplied_sine(self) -> Module:
        class MultipliedSine(_multiplied_activation):
            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return torch.sin(multiplied)

        config = self.configs.setdefault("multiplied_sine", {})
        config.setdefault("ratio", 10.0)
        return MultipliedSine(**config)

    @property
    def multiplied_tanh(self) -> Module:
        class MultipliedTanh(_multiplied_activation):
            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return torch.tanh(multiplied)

        return MultipliedTanh(**self.configs.setdefault("multiplied_tanh", {}))

    @property
    def multiplied_sigmoid(self) -> Module:
        class MultipliedSigmoid(_multiplied_activation):
            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return torch.sigmoid(multiplied)

        return MultipliedSigmoid(**self.configs.setdefault("multiplied_sigmoid", {}))

    @property
    def multiplied_softmax(self) -> Module:
        class MultipliedSoftmax(_multiplied_activation):
            def __init__(self, ratio: float, dim: int = 1, trainable: bool = True):
                super().__init__(ratio, trainable)
                self.dim = dim

            def _core(self, multiplied: torch.Tensor) -> torch.Tensor:
                return F.softmax(multiplied, dim=self.dim)

        return MultipliedSoftmax(**self.configs.setdefault("multiplied_softmax", {}))

    @property
    def cup_masked(self) -> Module:
        class CupMasked(Module):
            def __init__(
                self,
                bias: float = 2.0,
                ratio: float = 2.0,
                retain_sign: bool = False,
                trainable: bool = True,
            ):
                super().__init__()
                sigmoid_kwargs = {"ratio": ratio, "trainable": trainable}
                self.sigmoid = Activations.make("multiplied_sigmoid", sigmoid_kwargs)
                bias = math.log(math.exp(bias) - 1.0)
                bias_ = torch.tensor([bias], dtype=torch.float32)
                self.bias = bias_ if not trainable else nn.Parameter(bias_)
                self.retain_sign = retain_sign
                self.trainable = trainable

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                net_abs = net.abs()
                bias = F.softplus(self.bias)
                cup_mask = self.sigmoid(net_abs - bias)
                masked = net_abs * cup_mask
                if not self.retain_sign:
                    return masked
                return masked * torch.sign(net)

            def extra_repr(self) -> str:
                bias_str = f"(bias): {self.bias.item()}"
                positive_str = f"(positive): {not self.retain_sign}"
                trainable_str = f"(trainable): {self.trainable}"
                return f"{bias_str}\n{positive_str}\n{trainable_str}"

        return CupMasked(**self.configs.setdefault("cup_masked", {}))

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
        return cls({name: config}).module(name)


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
            ones_np = np.tile(ones_np, 2 ** i)
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
        self.eps: torch.Tensor
        self.exp: torch.Tensor
        self.alpha: Union[torch.Tensor, nn.Parameter]
        self.beta: Union[torch.Tensor, nn.Parameter]
        self.gamma: Union[torch.Tensor, nn.Parameter]
        self.max_ratio: Union[torch.Tensor, nn.Parameter]
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

    def forward(self, w: torch.Tensor) -> torch.Tensor:
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


class Attention(Module, WithRegister):
    d: Dict[str, Type["Attention"]] = attentions

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
            self.k_dim = k_dim or input_dim
            self.v_dim = v_dim or input_dim
        else:
            if k_dim is not None and k_dim != input_dim:
                raise ValueError("self attention is used but `k_dim` != `input_dim`")
            if v_dim is not None and v_dim != input_dim:
                raise ValueError("self attention is used but `v_dim` != `input_dim`")
            self.k_dim = self.v_dim = input_dim
        self.embed_dim = embed_dim or input_dim

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("`embed_dim` must be divisible by `num_heads`")

        def _warn(prefix: str) -> None:
            print(
                f"{WARNING_PREFIX}self attention is used so `{prefix}_linear_config` "
                "will be ignored, please use `in_linear_config` instead"
            )

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

    def _to_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.head_dim)

    def _get_weights(self, raw_weights: Tensor) -> Tensor:
        # in most cases the softmax version is good enough
        return F.softmax(raw_weights, dim=-1)

    def _weights_callback(self, weights: Tensor) -> Tensor:
        return weights

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        mask: Optional[Tensor] = None,
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
        # B * N_head, Sq, Sk -> B * N_head, Sq, Sk
        weights = self._get_weights(raw_weights)
        if 0.0 < self.dropout < 1.0:
            weights = F.dropout(weights, self.dropout, self.training)
        weights = self._weights_callback(weights)
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
        q_linear_config: Optional[Dict[str, Any]] = None,
        k_linear_config: Optional[Dict[str, Any]] = None,
        v_linear_config: Optional[Dict[str, Any]] = None,
        in_linear_config: Optional[Dict[str, Any]] = None,
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
            q_linear_config=q_linear_config,
            k_linear_config=k_linear_config,
            v_linear_config=v_linear_config,
            in_linear_config=in_linear_config,
            out_linear_config=out_linear_config,
        )
        mask = np.zeros([seq_len, seq_len], dtype=np.float32)
        for i in range(1, seq_len):
            np.fill_diagonal(mask[i:], i ** 2)
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
    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return F.normalize(net, dim=1)


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
        config = {}
        if norm_type == "batch":
            config = {"affine": True, "track_running_stats": True}
        elif norm_type == "instance":
            config = {"affine": False, "track_running_stats": False}
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


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return x + self.module(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, *dims: int, module: nn.Module, norm_type: str = "layer_norm"):
        super().__init__()
        self.norms = nn.ModuleList([])
        for dim in dims:
            self.norms.append(NormFactory(norm_type).make(dim))
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
        return self.module(*x_list, **kwargs).output


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


class MappingBase(Module, WithRegister):
    d: Dict[str, Type["MappingBase"]] = mapping_dict

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
            self.activation = Activations.make(activation, activation_config)
        use_dropout = 0.0 < dropout < 1.0
        self.dropout = None if not use_dropout else Dropout(dropout)

    @property
    def weight(self) -> Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return self.linear.bias

    def forward(self, net: Tensor, *, reuse: bool = False) -> Tensor:
        net = self.linear(net)
        if self.bn is not None:
            net = self.bn(net)
        if self.activation is not None:
            net = self.activation(net)
        if self.dropout is not None:
            net = self.dropout(net, reuse=reuse)
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


class _SkipConnectBlock(MappingBase, metaclass=ABCMeta):
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

    def forward(self, net: Tensor) -> Tensor:
        linear = self.linear_mapping(net)
        nonlinear = self.nonlinear_mapping(net)
        return self._skip_connect(net, linear, nonlinear)

    @abstractmethod
    def _skip_connect(self, net: Tensor, linear: Tensor, nonlinear: Tensor) -> Tensor:
        pass


@MappingBase.register("res")
class ResBlock(_SkipConnectBlock):
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
        super().__init__(
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
        self.res_linear = Linear(
            latent_dim,
            latent_dim,
            bias=True if bias is None else bias,
            pruner_config=pruner_config,
            init_method=init_method,
            **kwargs,
        )
        self.res_bn = None if not batch_norm else BN(latent_dim)
        self.res_dropout = Dropout(dropout)
        if activation is None:
            self.res_activation: Optional[Module] = None
        else:
            activation_config = kwargs.setdefault("activation_config", None)
            self.res_activation = Activations.make(activation, activation_config)

    def _skip_connect(self, net: Tensor, linear: Tensor, nonlinear: Tensor) -> Tensor:
        net = linear + self.res_linear(nonlinear)
        if self.res_bn is not None:
            net = self.res_bn(net)
        net = self.res_dropout(net)
        if self.res_activation is None:
            return net
        return self.res_activation(net)


@MappingBase.register("highway")
class HighwayBlock(_SkipConnectBlock):
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
        super().__init__(
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

    def _skip_connect(self, net: Tensor, linear: Tensor, nonlinear: Tensor) -> Tensor:
        gate = self.sigmoid(self.gate_linear(net))
        return gate * nonlinear + (1.0 - gate) * linear


# convolutions


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
        padding: Any = "reflection",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        gain: float = math.sqrt(2.0),
    ):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        self.kernel_size = kernel_size
        self.reflection_pad = None
        if padding == "reflection":
            padding = 0
            reflection_padding: Any = kernel_size // 2
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
        # initialize
        with torch.no_grad():
            # nn.init.normal_(self.weight.data, 0.0, 0.02)
            nn.init.xavier_normal_(self.weight.data, gain / math.sqrt(2.0))
            if self.bias is not None:
                self.bias.zero_()

    def _same_padding(self, size: int) -> int:
        stride = self.stride
        dilation = self.dilation
        return ((size - 1) * (stride - 1) + dilation * (self.kernel_size - 1)) // 2

    def forward(self, x: Tensor, style: Optional[Tensor] = None) -> Tensor:
        b, c, *hw = x.shape
        # padding
        padding = self.padding
        if self.padding == "same":
            padding = tuple(map(self._same_padding, hw))
        if self.reflection_pad is not None:
            x = self.reflection_pad(x)
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
            w = w[None, ...] * (style[..., None, :, None, None] + 1.0)
            # prepare for group convolution
            bias = None
            groups = b
            x = x.view(1, -1, *hw)  # 1, b*in, h, w
            w = w.view(b * self.out_c, *w.shape[2:])  # b*out, in, wh, ww
        if self.demodulate:
            w = w * torch.rsqrt(w.pow(2).sum([-3, -2, -1], keepdim=True) + 1e-8)
        # if style is provided, we should view the results back
        x = F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=groups,
        )
        if style is None:
            return x
        return x.view(-1, self.out_c, *hw)

    def extra_repr(self) -> str:
        return (
            f"{self.in_c}, {self.out_c}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"demodulate={self.demodulate}"
        )


def upscale(x: torch.Tensor, factor: float) -> torch.Tensor:
    return F.interpolate(x, scale_factor=factor, recompute_scale_factor=True)  # type: ignore


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
        padding: Union[int, str] = "reflection",
        transform_kernel: bool = False,
        bias: bool = True,
        demodulate: bool = False,
        factor: float = None,
    ):
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
        self.factor = factor

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if self.factor is not None:
            x = upscale(x, factor=self.factor)
        return super().forward(x, y)


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
    activation: Optional[Module] = None,
    **conv2d_kwargs: Any,
) -> List[Module]:
    blocks: List[Module] = [
        Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            demodulate=demodulate,
            **conv2d_kwargs,
        )
    ]
    if not demodulate:
        factory = NormFactory(norm_type)
        factory.inject_to(out_channels, norm_kwargs or {}, blocks)
    if activation is not None:
        blocks.append(activation)
    return blocks
