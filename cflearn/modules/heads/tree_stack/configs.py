from typing import Any
from typing import Dict

from ..base import HeadConfigs


@HeadConfigs.register("tree_stack", "default")
class DefaultTreeStackConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {"num_blocks": 3, "dndf_config": {}, "out_dndf_config": {}}


@HeadConfigs.register("tree_stack", "linear")
class LinearTreeStackConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {"num_blocks": 0, "dndf_config": {}, "out_dndf_config": {}}


__all__ = [
    "DefaultTreeStackConfig",
    "LinearTreeStackConfig",
]
