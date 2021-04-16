import copy
import math
import torch

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
from functools import partial
from torch.nn import Module
from torch.nn import ModuleList
from cftool.misc import update_dict
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict

from ..types import tensor_dict_type
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


class MappingBase(Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global mapping_dict
        return register_core(name, mapping_dict)


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
        if norm_type == "batch":
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
