from typing import Any
from typing import Dict

from ....configs import Configs


@Configs.register("transformer", "default")
class DefaultTransformerConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {
            "num_heads": 4,
            "num_layers": 2,
            "latent_dim": 256,
            "input_linear_config": {"bias": False},
            "transformer_layer_config": {"latent_dim": 1024},
        }


__all__ = ["DefaultTransformerConfig"]
