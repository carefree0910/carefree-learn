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


loss_type = Type[LossProtocol]
metric_type = Type[MetricProtocol]
pipeline_type = Type[PipelineProtocol]
processor_type = Type[Processor]


def register_processor(name: str) -> Callable[[processor_type], processor_type]:
    return Processor.register(name)  # type: ignore


def register_loss(name: str) -> Callable[[loss_type], loss_type]:
    return LossProtocol.register(name)


def register_metric(name: str) -> Callable[[metric_type], metric_type]:
    return MetricProtocol.register(name)


def register_pipeline(name: str) -> Callable[[pipeline_type], pipeline_type]:
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
