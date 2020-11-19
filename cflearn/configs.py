from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from typing import Optional
from cftool.misc import update_dict
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin


configs_dict: Dict[str, Dict[str, Type["Configs"]]] = {}


class Configs(ABC, LoggingMixin):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.config = config

    @abstractmethod
    def get_default(self) -> Dict[str, Any]:
        pass

    def pop(self) -> Dict[str, Any]:
        default = self.get_default()
        if self.config is None:
            return default
        return update_dict(shallow_copy_dict(self.config), default)

    def setdefault(self, key: str, value: Any) -> Any:
        if self.config is None:
            self.config = {key: value}
            return value
        return self.config.setdefault(key, value)

    @classmethod
    def register(cls, scope: str, name: str) -> Callable[[Type], Type]:
        global configs_dict

        def before(cls_: Type) -> None:
            cls_.name = name

        return register_core(
            name,
            configs_dict.setdefault(scope, {}),
            before_register=before,
        )

    @classmethod
    def get(cls, scope: str, name: str, **kwargs: Any) -> "Configs":
        return configs_dict[scope][name](kwargs)


__all__ = ["configs_dict", "Configs"]
