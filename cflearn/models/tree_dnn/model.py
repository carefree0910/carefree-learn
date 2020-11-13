import logging

from typing import Any
from typing import Dict

from ..base import ModelBase


@ModelBase.register("tree_dnn")
@ModelBase.register_pipe("dndf")
@ModelBase.register_pipe("fcnn", transform="embedding")
class TreeDNN(ModelBase):
    def _preset_config(self) -> None:
        mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("pruner_config", {})
        self.config.setdefault("default_encoding_method", ["embedding", "one_hot"])


@ModelBase.register("tree_stack")
@ModelBase.register_pipe("tree_stack")
class TreeStack(ModelBase):
    def define_pipe_configs(self) -> None:
        cfg = self.get_core_config(self)
        self.define_head_config("tree_stack", cfg)

    @property
    def output_probabilities(self) -> bool:
        return True

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        cfg = ModelBase.get_core_config(instance)
        warn_num_blocks = instance.config.get("warn_num_blocks", True)
        num_blocks = instance.config.setdefault("num_blocks", 3)
        if warn_num_blocks and num_blocks <= 0:
            instance.log_msg(  # type: ignore
                "`num_blocks` is 0 in TreeStack, it will be equivalent to TreeLinear",
                prefix=instance.warning_prefix,
                verbose_level=2,
                msg_level=logging.WARNING,
            )
        dndf_config = instance.config.setdefault("dndf_config", {})
        out_dndf_config = instance.config.setdefault("out_dndf_config", {})
        cfg["num_blocks"] = num_blocks
        cfg["dndf_config"] = dndf_config
        cfg["out_dndf_config"] = out_dndf_config
        return cfg

    def _init_config(self) -> None:
        super()._init_config()
        self._loss_config["input_logits"] = False


@TreeStack.register("tree_linear")
class TreeLinear(TreeStack):
    def _preset_config(self) -> None:
        super()._preset_config()
        self.config["num_blocks"] = 0
        self.config["warn_num_blocks"] = False


__all__ = ["TreeDNN", "TreeStack", "TreeLinear"]
