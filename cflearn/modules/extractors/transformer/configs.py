from typing import Any
from typing import Dict

from ....configs import Configs


@Configs.register("transformer", "default")
class DefaultTransformerConfig(Configs):
    def get_default(self) -> Dict[str, Any]:
        return {
            "num_heads": 4,
            "num_layers": 3,
            "latent_dim": 32,
            "dropout": 0.1,
            "norm_type": "layer_norm",
            "attention_type": "decayed",
            "encoder_type": "basic",
            "input_linear_config": {"bias": False},
            "layer_config": {"latent_dim": 128},
            "encoder_config": {},
            "use_head_token": True,
            "final_norm_type": "layer_norm",
        }


__all__ = ["DefaultTransformerConfig"]
