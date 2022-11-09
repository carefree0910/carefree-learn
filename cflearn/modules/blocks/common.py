import copy
import torch

import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Iterator
from typing import Optional
from functools import wraps
from torch.nn import Module
from torch.nn import ModuleList
from cftool.types import tensor_dict_type


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


class ApplyTanhMixin:
    apply_tanh: bool

    def postprocess(self, net: Tensor, apply_tanh: Optional[bool]) -> Tensor:
        if apply_tanh is None:
            apply_tanh = self.apply_tanh
        if apply_tanh:
            net = torch.tanh(net)
        return net


__all__ = [
    "get_clones",
    "reuse_fn",
    "Lambda",
    "EMA",
    "MTL",
    "ApplyTanhMixin",
]
