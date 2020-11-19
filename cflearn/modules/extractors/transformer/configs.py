from typing import Any
from typing import Dict

from ....configs import Configs


@Configs.register("transformer", "default")
class DefaultTransformerConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {
            "num_heads": 8,
            "num_layers": 6,
            "latent_dim": 256,
            "norm": None,
            "input_linear_config": {"bias": False},
            "transformer_layer_config": {},
        }


__all__ = ["DefaultTransformerConfig"]
