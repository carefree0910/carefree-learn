import torch

from torch import nn
from typing import Any
from typing import Callable
from typing import Iterator
from cftool.types import tensor_dict_type


class DropNoGradStatesMixin:
    def state_dict(
        self,
        *,
        destination: Any = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> tensor_dict_type:
        states = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)  # type: ignore
        for key, _ in self.named_buffers():  # type: ignore
            if states.pop(key, None) is None:
                states.pop(f"core.{key}")
        for key, value in self.named_parameters():  # type: ignore
            if not value.requires_grad:
                if states.pop(key, None) is None:
                    states.pop(f"core.{key}")
        return states

    def load_state_dict(
        self,
        state_dict: tensor_dict_type,
        strict: bool = True,
    ) -> None:
        with torch.no_grad():
            for key, value in self.named_parameters():  # type: ignore
                if value.requires_grad:
                    loaded_value = state_dict.get(key)
                    if strict and loaded_value is None:
                        raise ValueError(f"value for '{key}' is missing")
                    value.data.copy_(loaded_value)
