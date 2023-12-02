from torch import nn
from typing import Any
from typing import Dict
from typing import Union
from typing import Callable
from typing import Optional

from ..common import TModule
from ..common import PrefixModules


ml_modules = PrefixModules("ml_modules")


def register_ml_module(name: str, **kwargs: Any) -> Callable[[TModule], TModule]:
    return ml_modules.register(name, **kwargs)


def build_ml_module(
    name: str,
    *,
    config: Optional[Union[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> nn.Module:
    return ml_modules.build(name, config=config, **kwargs)


__all__ = [
    "ml_modules",
    "register_ml_module",
    "build_ml_module",
]
