from typing import *

from ..base import ModelBase
from ..fcnn import FCNN


@ModelBase.register("rnn")
@ModelBase.register_pipe("rnn", head="fcnn")
class RNN(ModelBase):
    def define_pipe_configs(self) -> None:
        cfg = self.get_core_config(self)
        self.define_head_config("rnn", cfg["fcnn"])
        self.define_extractor_config("rnn", cfg["rnn"])

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        rnn_cfg = instance.config.setdefault("rnn_config", {})
        rnn_cfg.setdefault("cell", "GRU")
        rnn_cfg.setdefault("num_layers", 1)
        cell_config = rnn_cfg.setdefault("cell_config", {})
        cell_config["batch_first"] = True
        cell_config.setdefault("hidden_size", 256)
        cell_config.setdefault("bidirectional", False)
        fcnn_cfg = FCNN.get_core_config(instance)
        return {"fcnn": fcnn_cfg, "rnn": rnn_cfg}


__all__ = ["RNN"]
