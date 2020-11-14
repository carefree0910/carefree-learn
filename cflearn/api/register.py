from typing import *
from cftool.ml.utils import register_metric
from cfdata.tabular.processors.base import Processor

from ..modules import *
from ..misc.configs import Configs
from ..misc.toolkit import Initializer
from ..models.base import ModelBase
from ..modules.heads.base import HeadConfigs


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_processor(name: str) -> Callable[[Type], Type]:
    return Processor.register(name)


def register_model(name: str) -> Callable[[Type], Type]:
    return ModelBase.register(name)


def register_pipe(
    key: str,
    *,
    transform: str = "default",
    extractor: Optional[str] = None,
    head: Optional[str] = None,
    extractor_config: str = "default",
    head_config: str = "default",
    extractor_meta_scope: Optional[str] = None,
    head_meta_scope: Optional[str] = None,
) -> Callable[[Type], Type]:
    return ModelBase.register_pipe(
        key,
        transform=transform,
        extractor=extractor,
        head=head,
        extractor_config=extractor_config,
        head_config=head_config,
        extractor_meta_scope=extractor_meta_scope,
        head_meta_scope=head_meta_scope,
    )


def register_config(scope: str, name: str) -> Callable[[Type], Type]:
    return Configs.register(scope, name)


def register_head_config(scope: str, name: str) -> Callable[[Type], Type]:
    return HeadConfigs.register(scope, name)


__all__ = [
    "register_pipe",
    "register_model",
    "register_config",
    "register_head_config",
    "register_metric",
    "register_optimizer",
    "register_scheduler",
    "register_initializer",
    "register_processor",
    "Initializer",
    "Processor",
]
