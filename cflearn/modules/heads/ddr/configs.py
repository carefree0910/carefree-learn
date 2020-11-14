from typing import Any
from typing import Dict

from ..base import HeadConfigs


@HeadConfigs.register("ddr", "default")
class DefaultDDRConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        return {
            "fetch_q": True,
            "fetch_cdf": True,
            "num_layers": 1,
            "num_blocks": 2,
            "latent_dim": 512,
        }


__all__ = ["DefaultDDRConfig"]
