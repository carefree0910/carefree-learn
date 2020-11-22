from typing import *
from cftool.misc import shallow_copy_dict
from cftool.ml.utils import register_metric
from cfdata.tabular.processors.base import Processor

from ..modules import *
from ..configs import Configs
from ..misc.toolkit import Initializer
from ..models.base import ModelBase
from ..modules.heads import HeadBase
from ..modules.heads import HeadConfigs
from ..modules.extractors import ExtractorBase


def register_initializer(name: str) -> Callable[[Callable], Callable]:
    def _register(f: Callable) -> Callable:
        Initializer.add_initializer(f, name)
        return f

    return _register


def register_processor(name: str) -> Callable[[Type], Type]:
    return Processor.register(name)


def register_model(name: str) -> Callable[[Type], Type]:
    return ModelBase.register(name)


def register_extractor(name: str) -> Callable[[Type], Type]:
    return ExtractorBase.register(name)


def register_head(name: str) -> Callable[[Type], Type]:
    return HeadBase.register(name)


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


def _register_config(
    base: Type[Configs],
    scope: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[[Type], Type]]:
    if config is None:
        return base.register(scope, name)

    def register(config_: Dict[str, Any]) -> None:
        @base.register(scope, name)
        class _(base):  # type: ignore
            def get_default(self) -> Dict[str, Any]:
                return shallow_copy_dict(config_)

    register(config)
    return None


def register_config(
    scope: str,
    name: str,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[[Type], Type]]:
    return _register_config(Configs, scope, name, config)  # type: ignore


def register_head_config(
    scope: str,
    name: str,
    *,
    head_config: Optional[Dict[str, Any]] = None,
) -> Optional[Callable[[Type], Type]]:
    return _register_config(HeadConfigs, scope, name, head_config)  # type: ignore


__all__ = [
    "register_pipe",
    "register_head",
    "register_extractor",
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
