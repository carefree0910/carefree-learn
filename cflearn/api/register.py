from cftool.ml.utils import register_metric
from cfdata.tabular.processors.base import Processor

from ..modules import *
from ..misc.toolkit import Initializer


def register_initializer(name):
    def _register(f):
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_processor(name):
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
