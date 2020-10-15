from typing import *
from cftool.ml.utils import register_metric
from cfdata.tabular.processors.base import Processor

from ..modules import *
from ..misc.toolkit import Initializer


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_processor(name: str) -> Callable[[Type], Type]:
    return Processor.register(name)


__all__ = [
    "register_metric",
    "register_optimizer",
    "register_scheduler",
    "register_initializer",
    "register_processor",
    "Initializer",
    "Processor",
]
