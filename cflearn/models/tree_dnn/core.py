import torch
import logging

from typing import *
from cfdata.tabular import TabularData

from ..fcnn.core import FCNN
from ...bases import SplitFeatures
from ...misc.toolkit import tensor_dict_type
from ...modules.blocks import *


@FCNN.register("tree_dnn")
class TreeDNN(FCNN):
    def __init__(
        self,
        config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super(FCNN, self).__init__(config, tr_data, device)
        encoding_dims = self.encoding_dims
        embedding_dims = encoding_dims.get("embedding", 0)
        one_hot_dims = encoding_dims.get("one_hot", 0)
        # fc
        if not self._numerical_columns:
            self._use_embedding_for_fc = True
            self._use_one_hot_for_fc = "one_hot" in self._default_encoding_method
        fc_in_dim = self.merged_dim
        if not self._use_embedding_for_fc:
            fc_in_dim -= embedding_dims
        if not self._use_one_hot_for_fc:
            fc_in_dim -= one_hot_dims
        self.config["fc_in_dim"] = fc_in_dim
        self._init_fcnn()
        # dndf
        if self._dndf_config is None:
            self.log_msg(
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
                dndf_input_dim -= embedding_dims
            if not self._use_one_hot_for_dndf:
                dndf_input_dim -= one_hot_dims
            self.dndf = DNDF(dndf_input_dim, self._fc_out_dim, **self._dndf_config)

    def _preset_config(self, tr_data: TabularData):
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
            return numerical
        categorical = split_result.categorical
        if not categorical:
            return numerical
        if not use_one_hot:
            embedding = categorical["embedding"]
            if numerical is None:
                return embedding
            return torch.cat([numerical, embedding], dim=1)
        if not use_embedding:
            one_hot = categorical["one_hot"]
            if numerical is None:
                return one_hot
            return torch.cat([numerical, one_hot], dim=1)

    def forward(self, batch: tensor_dict_type, **kwargs) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch, return_all_encodings=True)
        # fc
        fc_net = self._merge(
            split_result, self._use_embedding_for_fc, self._use_one_hot_for_fc
        )
        fc_net = self.mlp(fc_net)
        # dndf
        if self.dndf is None:
            return {"predictions": fc_net}
        dndf_net = self._merge(
            split_result, self._use_embedding_for_dndf, self._use_one_hot_for_dndf
        )
        dndf_net = self.dndf(dndf_net)
        return {"predictions": fc_net + dndf_net}


__all__ = ["TreeDNN"]
