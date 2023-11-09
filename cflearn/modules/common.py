import json
import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Iterator
from typing import Optional
from torch.nn import Module
from cftool.misc import update_dict
from cftool.misc import safe_execute
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.types import tensor_dict_type


# managements


TModule = TypeVar("TModule", bound=Type[Module])

module_dict: Dict[str, Type["Module"]] = {}


def register_module(name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
    return register_core(name, module_dict, **kwargs)


def build_module(
    name: str,
    *,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Module:
    if config is None:
        kw = shallow_copy_dict(kwargs)
    else:
        if isinstance(config, dict):
            kw = config
        else:
            with open(config, "r") as f:
                kw = json.load(f)
        kw = shallow_copy_dict(kw)
        update_dict(shallow_copy_dict(kwargs), kw)
    return safe_execute(module_dict[name], kw)


class PrefixModules:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def has(self, name: str) -> bool:
        return self._prefix_name(name) in module_dict

    def get(self, name: str) -> Optional[Type[Module]]:
        return module_dict.get(self._prefix_name(name))

    def register(self, name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
        return register_module(self._prefix_name(name), **kwargs)

    def build(
        self,
        name: str,
        *,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Module:
        return build_module(self._prefix_name(name), config=config, **kwargs)

    def _prefix_name(self, name: str) -> str:
        return f"{self._prefix}.{name}"


# common building blocks


class Lambda(Module):
    def __init__(self, fn: Callable, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.fn = fn

    def extra_repr(self) -> str:
        return "" if self.name is None else self.name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class EMA(Module):
    def __init__(
        self,
        decay: float,
        named_parameters: List[Tuple[str, nn.Parameter]],
        *,
        use_num_updates: bool = False,
    ):
        super().__init__()
        self._cache: tensor_dict_type = {}
        self._decay = decay
        self._named_parameters = named_parameters
        for name, param in self.tgt_params:
            self.register_buffer(name, param.data.clone())
        num_updates = torch.tensor(0 if use_num_updates else -1, dtype=torch.int)
        self.register_buffer("num_updates", num_updates)

    @property
    def tgt_params(self) -> Iterator[Tuple[str, nn.Parameter]]:
        return map(
            lambda pair: (pair[0].replace(".", "_"), pair[1]),
            self._named_parameters,
        )

    def forward(self) -> None:
        if not self.training:
            raise ValueError("should not update `EMA` at inference stage")
        if self.num_updates < 0:
            decay = self._decay
        else:
            self.num_updates += 1
            decay = min(self._decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.tgt_params:
            ema_attr = getattr(self, name)
            ema = (1.0 - decay) * param.data + decay * ema_attr
            setattr(self, name, ema.clone())

    def train(self, mode: bool = True) -> "EMA":
        super().train(mode)
        if mode:
            for name, param in self.tgt_params:
                cached = self._cache.pop(name, None)
                if cached is not None:
                    param.data = cached
        else:
            for name, param in self.tgt_params:
                if name not in self._cache:
                    self._cache[name] = param.data
                param.data = getattr(self, name).clone()
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


# common structures


class Residual(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, net: Tensor, **kwargs: Any) -> Tensor:
        return net + self.module(net, **kwargs)


def zero_module(module: Module) -> Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(n: int, *args: Any, **kwargs: Any) -> Module:
    if n == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif n == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif n == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {n}")


__all__ = [
    "module_dict",
    "register_module",
    "build_module",
    "PrefixModules",
    "Lambda",
    "Residual",
    "zero_module",
    "avg_pool_nd",
]
