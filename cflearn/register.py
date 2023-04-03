from typing import Callable

from .schema import IMetric
from .misc.toolkit import Initializer


def register_initializer(name: str) -> Callable:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_metric(name: str, *, allow_duplicate: bool = False) -> Callable:
    return IMetric.register(name, allow_duplicate=allow_duplicate)


__all__ = [
    "register_initializer",
    "register_metric",
]
