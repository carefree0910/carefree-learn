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
    @property
    def output_probabilities(self) -> bool:
        return True

    def _init_config(self) -> None:
        super()._init_config()
        self._loss_config["input_logits"] = False


@TreeStack.register("tree_linear")
@ModelBase.register_pipe("tree_stack", head_config="linear")
class TreeLinear(TreeStack):
    pass


__all__ = ["TreeDNN", "TreeStack", "TreeLinear"]
