from typing import Any
from typing import Dict

from ....misc.configs import Configs


@Configs.register("identity", "default")
class DefaultIdentityConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {}


__all__ = ["DefaultIdentityConfig"]
