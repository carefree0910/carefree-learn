from typing import Any
from typing import Dict

from ..base import HeadConfigs


@HeadConfigs.register("linear", "default")
class DefaultLinearConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {"linear_config": {}}


__all__ = ["DefaultLinearConfig"]
