import torch
import logging

import numpy as np

from typing import Any
from typing import Dict
from typing import Optional
from cfdata.tabular import TabularData

from ..base import SplitFeatures
from ..fcnn.core import FCNN
from ...types import tensor_dict_type
from ...modules.blocks import *


@FCNN.register("tree_dnn")
class TreeDNN(FCNN):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_data: TabularData,
        cv_data: TabularData,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
    ):
        self.dndf: Optional[DNDF]
        self._dndf_config: Dict[str, Any]
        super(FCNN, self).__init__(
            pipeline_config,
            tr_data,
            cv_data,
            tr_weights,
            cv_weights,
            device,
        )
        encoding_dims = self.encoding_dims
        embedding_dims = encoding_dims.get("embedding", 0)
        one_hot_dims = encoding_dims.get("one_hot", 0)
        # fc
        if self.use_fcnn:
            if not self._numerical_columns:
                self._use_embedding_for_fc = True
                self._use_one_hot_for_fc = "one_hot" in self._default_encoding_method
            fc_in_dim = self.merged_dim
            if not self._use_embedding_for_fc:
                fc_in_dim -= embedding_dims * self.num_history
            if not self._use_one_hot_for_fc:
                fc_in_dim -= one_hot_dims * self.num_history
            self.config["fc_in_dim"] = fc_in_dim
            self._init_fcnn()
        # dndf
        if self._dndf_config is None:
            self.log_msg(  # type: ignore
                "DNDF is not used in TreeDNN, it will be equivalent to FCNN",
                prefix=self.warning_prefix,
                verbose_level=2,
                msg_level=logging.WARNING,
            )
            self.dndf = None
        else:
            self._dndf_config["is_regression"] = tr_data.is_reg
            self._dndf_config.setdefault("tree_proj_config", None)
            if not self._numerical_columns:
                self._use_embedding_for_dndf = True
                self._use_one_hot_for_dndf = "one_hot" in self._default_encoding_method
            dndf_input_dim = self.merged_dim
            if not self._use_embedding_for_dndf:
                dndf_input_dim -= embedding_dims * self.num_history
            if not self._use_one_hot_for_dndf:
                dndf_input_dim -= one_hot_dims * self.num_history
            self.dndf = DNDF(dndf_input_dim, self._fc_out_dim, **self._dndf_config)

    def _preset_config(self) -> None:
        mapping_configs = self.config.setdefault("mapping_configs", {})
        if isinstance(mapping_configs, dict):
            mapping_configs.setdefault("pruner_config", {})
        self._use_embedding_for_fc = self.config.setdefault(
            "use_embedding_for_fc", True
        )
        self._use_one_hot_for_fc = self.config.setdefault("use_one_hot_for_fc", False)
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
        if self._use_one_hot_for_fc or self._use_one_hot_for_dndf:
            default_encoding_method.append("one_hot")
        self.config.setdefault("default_encoding_method", default_encoding_method)

    def _init_config(self) -> None:
        super()._init_config()
        self.use_fcnn = self.config.setdefault("use_fcnn", True)
        if not self.use_fcnn and self._dndf_config is None:
            raise ValueError("either `fcnn` or `dndf` should be used")

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
        assert isinstance(categorical, dict)
        if not use_one_hot:
            embedding = categorical["embedding"]
            if numerical is None:
                return embedding
            return torch.cat([numerical, embedding], dim=1)
        assert not use_embedding
        one_hot = categorical["one_hot"]
        if numerical is None:
            return one_hot
        return torch.cat([numerical, one_hot], dim=1)

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch, return_all_encodings=True)
        # fc
        if not self.use_fcnn:
            fc_net = None
        else:
            fc_net = self._merge(
                split_result,
                self._use_embedding_for_fc,
                self._use_one_hot_for_fc,
            )
            if self.tr_data.is_ts:
                fc_net = fc_net.view(fc_net.shape[0], -1)
            fc_net = self.mlp(fc_net)
        # dndf
        if self.dndf is None:
            assert fc_net is not None
            return {"predictions": fc_net}
        dndf_net = self._merge(
            split_result,
            self._use_embedding_for_dndf,
            self._use_one_hot_for_dndf,
        )
        if self.tr_data.is_ts:
            dndf_net = dndf_net.view(dndf_net.shape[0], -1)
        dndf_net = self.dndf(dndf_net)
        # merge
        if fc_net is None:
            return {"predictions": dndf_net}
        return {"predictions": fc_net + dndf_net}


@FCNN.register("tree_linear")
class TreeLinear(TreeDNN):
    @property
    def output_probabilities(self) -> bool:
        return True

    def _preset_config(self) -> None:
        super()._preset_config()
        self.config["use_fcnn"] = False

    def _init_config(self) -> None:
        super()._init_config()
        self._loss_config["input_logits"] = False
        self._fc_out_dim: int = self.config.get("fc_out_dim")
        self.out_dim = max(self.tr_data.num_classes, 1)
        if self._fc_out_dim is None:
            self._fc_out_dim = self.out_dim


__all__ = ["TreeDNN"]
