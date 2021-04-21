from typing import *
from cftool.ml.utils import register_metric
from cfdata.tabular.processors.base import Processor

from ..modules import *
from ..protocol import LossProtocol
from ..misc.toolkit import Initializer


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_processor(name: str) -> Callable[[Type], Type]:
    return Processor.register(name)


def register_loss(name: str) -> Callable[[Type], Type]:
    return LossProtocol.register(name)


__all__ = [
    "register_metric",
    "register_optimizer",
    "register_scheduler",
    "register_initializer",
    "register_processor",
    "register_loss",
    "Initializer",
    "Processor",
]
