import typing
import logging

from typing import Any
from typing import Dict

from ..base import ModelBase
from ..fcnn.model import FCNN


@ModelBase.register("tree_dnn")
@ModelBase.register_pipe("fcnn")
@ModelBase.register_pipe("dndf")
class TreeDNN(ModelBase):
    @staticmethod
    @typing.no_type_check
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        one_hot_dim = instance.one_hot_dim
        embedding_dim = instance.embedding_dim
        default_has_one_hot = "one_hot" in instance._default_encoding_method
        # fcnn
        if not instance._numerical_columns:
            instance._use_embedding_for_fcnn = True
            instance._use_one_hot_for_fcnn = default_has_one_hot
        fcnn_in_dim = instance.merged_dim
        if not instance._use_embedding_for_fcnn:
            fcnn_in_dim -= embedding_dim * instance.num_history
        if not instance._use_one_hot_for_fcnn:
            fcnn_in_dim -= one_hot_dim * instance.num_history
        instance.config["in_dim"] = fcnn_in_dim
        fcnn_cfg = FCNN.get_core_config(instance)
        # dndf
        if instance._dndf_config is None:
            instance.log_msg(  # type: ignore
                "DNDF is not used in TreeDNN, it will be equivalent to FCNN",
                prefix=instance.warning_prefix,
                verbose_level=2,
                msg_level=logging.WARNING,
            )
            dndf_input_dim = None
        else:
            instance._dndf_config["is_regression"] = instance.tr_data.is_reg
            instance._dndf_config.setdefault("tree_proj_config", None)
            if not instance._numerical_columns:
                instance._use_embedding_for_dndf = True
                instance._use_one_hot_for_dndf = default_has_one_hot
            dndf_input_dim = instance.merged_dim
            if not instance._use_embedding_for_dndf:
                dndf_input_dim -= embedding_dim * instance.num_history
            if not instance._use_one_hot_for_dndf:
                dndf_input_dim -= one_hot_dim * instance.num_history
        dndf_cfg = {
            "in_dim": dndf_input_dim,
            "out_dim": fcnn_cfg["out_dim"],
            "config": instance._dndf_config,
        }
        return {"fcnn_config": fcnn_cfg, "dndf_config": dndf_cfg}

    def define_pipe_configs(self) -> None:
        self.define_transform_config(
            "fcnn",
            {
                "one_hot": self._use_one_hot_for_fcnn,
                "embedding": self._use_embedding_for_fcnn,
            },
        )
        self.define_transform_config(
            "dndf",
            {
                "one_hot": self._use_one_hot_for_dndf,
                "embedding": self._use_embedding_for_dndf,
            },
        )
        cfg = self.get_core_config(self)
        self.define_head_config("fcnn", cfg["fcnn_config"])
        self.define_head_config("dndf", cfg["dndf_config"])

    def _preset_config(self) -> None:
        mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("pruner_config", {})
        fcnn_one_hot = self.config.setdefault("use_one_hot_for_fcnn", False)
        fcnn_embedding = self.config.setdefault("use_embedding_for_fcnn", True)
        self._dndf_config = self.config.setdefault("dndf_config", {})
        has_dndf = self._dndf_config is not None
        dndf_one_hot = self.config.setdefault("use_one_hot_for_dndf", has_dndf)
        dndf_embedding = self.config.setdefault("use_embedding_for_dndf", has_dndf)
        self._use_one_hot_for_fcnn = fcnn_one_hot
        self._use_embedding_for_fcnn = fcnn_embedding
        self._use_one_hot_for_dndf = dndf_one_hot
        self._use_embedding_for_dndf = dndf_embedding
        default_encoding_method = ["embedding"]
        if fcnn_one_hot or dndf_one_hot:
            default_encoding_method.append("one_hot")
        self.config.setdefault("default_encoding_method", default_encoding_method)


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
