import torch
import typing
import logging

import numpy as np

from typing import Any
from typing import Dict
from typing import Optional
from cfdata.tabular import DataLoader

from .core import TreeDNNCore
from .core import TreeStackCore
from ..base import ModelBase
from ..base import SplitFeatures
from ..fcnn.model import FCNN
from ...types import tensor_dict_type


@ModelBase.register("tree_dnn")
class TreeDNN(ModelBase):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__(
            pipeline_config,
            tr_loader,
            cv_loader,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        cfg = self.get_core_config(self)
        self.core = TreeDNNCore(**cfg)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    @staticmethod
    @typing.no_type_check
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        one_hot_dim = instance.one_hot_dim
        embedding_dim = instance.embedding_dim
        default_has_one_hot = "one_hot" in instance._default_encoding_method
        # fcnn
        if not instance._numerical_columns:
            instance._use_embedding_for_mlp = True
            instance._use_one_hot_for_mlp = default_has_one_hot
        mlp_in_dim = instance.merged_dim
        if not instance._use_embedding_for_mlp:
            mlp_in_dim -= embedding_dim * instance.num_history
        if not instance._use_one_hot_for_mlp:
            mlp_in_dim -= one_hot_dim * instance.num_history
        instance.config["in_dim"] = mlp_in_dim
        cfg = FCNN.get_core_config(instance)
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
        cfg["dndf_config"] = instance._dndf_config
        cfg["dndf_input_dim"] = dndf_input_dim
        return cfg

    def _preset_config(self) -> None:
        mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("pruner_config", {})
        self._use_embedding_for_mlp = self.config.setdefault(
            "use_embedding_for_fc", True
        )
        self._use_one_hot_for_mlp = self.config.setdefault("use_one_hot_for_fc", False)
        self._dndf_config = self.config.setdefault("dndf_config", {})
        self._use_embedding_for_dndf = self._use_one_hot_for_dndf = False
        if self._dndf_config is not None:
            self._use_embedding_for_dndf = self.config.setdefault(
                "use_embedding_for_dndf", True
            )
            self._use_one_hot_for_dndf = self.config.setdefault(
                "use_one_hot_for_dndf", True
            )
        default_encoding_method = ["embedding"]
        if self._use_one_hot_for_mlp or self._use_one_hot_for_dndf:
            default_encoding_method.append("one_hot")
        self.config.setdefault("default_encoding_method", default_encoding_method)

    @staticmethod
    def _merge(
        split_result: SplitFeatures,
        use_embedding: bool,
        use_one_hot: bool,
    ) -> torch.Tensor:
        if use_embedding and use_one_hot:
            return split_result.merge()
        numerical = split_result.numerical
        if not use_embedding and not use_one_hot:
            assert numerical is not None
            return numerical
        categorical = split_result.categorical
        if not categorical:
            assert numerical is not None
            return numerical
        if not use_one_hot:
            embedding = categorical.embedding
            assert embedding is not None
            if numerical is None:
                return embedding
            return torch.cat([numerical, embedding], dim=1)
        one_hot = categorical.one_hot
        assert not use_embedding and one_hot is not None
        if numerical is None:
            return one_hot
        return torch.cat([numerical, one_hot], dim=1)

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch, batch_indices, loader_name)
        # fcnn
        fcnn_net = self._merge(
            split_result,
            self._use_embedding_for_mlp,
            self._use_one_hot_for_mlp,
        )
        if self.tr_data.is_ts:
            fcnn_net = fcnn_net.view(fcnn_net.shape[0], -1)
        # dndf
        if self.core.dndf is None:
            dndf_net = None
        else:
            dndf_net = self._merge(
                split_result,
                self._use_embedding_for_dndf,
                self._use_one_hot_for_dndf,
            )
            if self.tr_data.is_ts:
                dndf_net = dndf_net.view(dndf_net.shape[0], -1)
        return {"predictions": self.core(fcnn_net, dndf_net)}


@ModelBase.register("tree_stack")
class TreeStack(ModelBase):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__(
            pipeline_config,
            tr_loader,
            cv_loader,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        cfg = self.get_core_config(self)
        self.core = TreeStackCore(**cfg)

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

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

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        return self.common_forward(self, batch, batch_indices, loader_name)


@TreeStack.register("tree_linear")
class TreeLinear(TreeStack):
    def _preset_config(self) -> None:
        super()._preset_config()
        self.config["num_blocks"] = 0
        self.config["warn_num_blocks"] = False


__all__ = ["TreeDNN", "TreeStack", "TreeLinear"]
