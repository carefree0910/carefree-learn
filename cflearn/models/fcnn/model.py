from typing import *

from ..base import ModelBase


@ModelBase.register("fcnn")
@ModelBase.register_pipe("fcnn")
class FCNN(ModelBase):
    def define_pipe_configs(self) -> None:
        cfg = self.get_core_config(self)
        self.define_head_config("fcnn", cfg)

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        cfg = ModelBase.get_core_config(instance)
        in_dim: int = cfg["in_dim"]
        if in_dim > 512:
            hidden_units = [1024, 1024]
        elif in_dim > 256:
            if len(instance.tr_data) >= 10000:
                hidden_units = [1024, 1024]
            else:
                hidden_units = [2 * in_dim, 2 * in_dim]
        else:
            num_tr_data = len(instance.tr_data)
            if num_tr_data >= 100000:
                hidden_units = [768, 768]
            elif num_tr_data >= 10000:
                hidden_units = [512, 512]
            else:
                hidden_dim = max(64 if num_tr_data >= 1000 else 32, 2 * in_dim)
                hidden_units = [hidden_dim, hidden_dim]
        hidden_units = instance.config.setdefault("hidden_units", hidden_units)
        mapping_configs = instance.config.setdefault("mapping_configs", {})
        fm_config = instance.config.setdefault("final_mapping_config", {})
        cfg.update(
            {
                "hidden_units": hidden_units,
                "mapping_configs": mapping_configs,
                "final_mapping_config": fm_config,
            }
        )
        return cfg


__all__ = ["FCNN"]
