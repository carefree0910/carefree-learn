from typing import Any
from typing import Dict

from ..base import ModelBase
from ..fcnn import FCNN


@ModelBase.register("transformer")
@ModelBase.register_pipe("transformer", head="fcnn", extractor="transformer")
class Transformer(ModelBase):
    def define_pipe_configs(self) -> None:
        cfg = self.get_core_config(self)
        self.define_head_config("transformer", cfg["fcnn"])
        self.define_extractor_config("transformer", cfg["transformer"])

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        in_dim = instance.tr_data.processed_dim
        transformer_cfg = instance.config.setdefault("transformer_config", {})
        latent_dim = transformer_cfg.setdefault("latent_dim", 256)
        transformer_cfg.setdefault("to_latent", latent_dim is not None)
        il_config = transformer_cfg.setdefault("input_linear_config", None)
        if il_config is not None:
            il_config.setdefault("bias", False)
        transformer_cfg.setdefault("num_layers", 6)
        transformer_cfg.setdefault("num_heads", 8)
        transformer_cfg.setdefault("norm", None)
        transformer_cfg.setdefault("transformer_layer_config", {})
        fcnn_cfg = FCNN.get_core_config(instance)
        return {"fcnn": fcnn_cfg, "transformer": transformer_cfg}


__all__ = ["Transformer"]
