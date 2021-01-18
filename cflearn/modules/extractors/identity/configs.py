from typing import Any
from typing import Dict

from ....configs import Configs


@Configs.register("identity", "default")
class DefaultIdentityConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {}


@Configs.register("identity_ts", "default")
class DefaultIdentityTSConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {}


__all__ = ["DefaultIdentityConfig", "DefaultIdentityTSConfig"]
