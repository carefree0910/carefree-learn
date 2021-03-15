from typing import Any
from typing import Dict

from ..base import HeadConfigs


@HeadConfigs.register("fcnn", "default")
class DefaultFCNNConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        in_dim = self.in_dim
        if in_dim > 512:
            hidden_units = [1024, 1024]
        elif in_dim > 256:
            if len(self.tr_data) >= 10000:
                hidden_units = [1024, 1024]
            else:
                hidden_units = [2 * in_dim, 2 * in_dim]
        else:
            num_tr_data = len(self.tr_data)
            if num_tr_data >= 100000:
                hidden_units = [768, 768]
            elif num_tr_data >= 10000:
                hidden_units = [512, 512]
            else:
                hidden_dim = max(64 if num_tr_data >= 1000 else 32, 2 * in_dim)
                hidden_units = [hidden_dim, hidden_dim]
        return {
            "hidden_units": hidden_units,
            "mapping_configs": {},
            "mapping_type": "basic",
            "final_mapping_config": {},
        }


@HeadConfigs.register("fcnn", "pruned")
class PrunedFCNNConfig(DefaultFCNNConfig):
    def get_default(self) -> Dict[str, Any]:
        config = super().get_default()
        config["mapping_configs"]["pruner_config"] = {}
        return config


__all__ = [
    "DefaultFCNNConfig",
]
