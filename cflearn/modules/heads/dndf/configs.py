from typing import Any
from typing import Dict

from ..base import HeadConfigs


@HeadConfigs.register("dndf", "default")
class DefaultDNDFConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {
            "dndf_config": {
                "is_regression": self.tr_data.is_reg,
                "tree_proj_config": None,
            }
        }


__all__ = [
    "DefaultDNDFConfig",
]
