from typing import Any
from typing import Callable
from torch.nn import Module

from ....common import TModule
from ....common import PrefixModules


# `condition_models` are rather 'independent' models, they can be used independently
condition_models = PrefixModules("condition_models")
# `specialized_condition_models` are specialized for conditioning, they cannot be used independently
specialized_condition_models = PrefixModules("specialized_condition_models")


def register_condition_model(name: str) -> Callable[[TModule], TModule]:
    return condition_models.register(name)


def register_specialized_condition_model(name: str) -> Callable[[TModule], TModule]:
    return specialized_condition_models.register(name)


def build_condition_model(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> Module:
    return condition_models.build(name, config=config, **kwargs)


def build_specialized_condition_model(
    name: str,
    *,
    config: Any = None,
    **kwargs: Any,
) -> Module:
    return specialized_condition_models.build(name, config=config, **kwargs)


__all__ = [
    "condition_models",
    "specialized_condition_models",
    "register_condition_model",
    "register_specialized_condition_model",
    "build_condition_model",
    "build_specialized_condition_model",
]
