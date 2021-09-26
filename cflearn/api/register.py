from typing import Type
from typing import Callable
from cfdata.tabular.processors.base import Processor

from ..modules import register_optimizer
from ..modules import register_scheduler
from ..pipeline import PipelineProtocol
from ..protocol import LossProtocol
from ..protocol import MetricProtocol
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


def register_metric(name: str) -> Callable[[Type], Type]:
    return MetricProtocol.register(name)


def register_pipeline(name: str) -> Callable[[Type], Type]:
    return PipelineProtocol.register(name)


__all__ = [
    "register_optimizer",
    "register_scheduler",
    "register_initializer",
    "register_processor",
    "register_loss",
    "register_metric",
    "register_pipeline",
    "Initializer",
    "Processor",
]
