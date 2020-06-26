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
    def __init__(self,
                 config: Dict[str, Any],
                 tr_data: TabularData,
                 device: torch.device):
        super(FCNN, self).__init__(config, tr_data, device)
        encoding_dims = self.encoding_dims
        # fc
        self._use_embedding_for_fc = self.config.setdefault("use_embedding_for_fc", True)
        self._use_one_hot_for_fc = self.config.setdefault("use_one_hot_for_fc", False)
        if not self._numerical_columns:
            self._use_embedding_for_fc = self._use_one_hot_for_fc = True
        fc_in_dim = self.merged_dim
        if not self._use_embedding_for_fc:
            fc_in_dim -= encoding_dims["embedding"]
        if not self._use_one_hot_for_fc:
            fc_in_dim -= encoding_dims["one_hot"]
        self._fc_in_dim = fc_in_dim
        self._init_fcnn()
        # dndf
        dndf_config = config.setdefault("dndf_config", {})
        if dndf_config is None:
            self.log_msg(
                "DNDF is not used in TreeDNN, it will be equivalent to FCNN",
                self.warning_prefix, 2, logging.WARNING
            )
            self.dndf = None
        else:
            dndf_config["is_regression"] = tr_data.is_reg
            dndf_config.setdefault("tree_proj_config", None)
            self._use_embedding_for_dndf = self.config.setdefault("use_embedding_for_dndf", True)
            self._use_one_hot_for_dndf = self.config.setdefault("use_one_hot_for_dndf", True)
            if not self._numerical_columns:
                self._use_one_hot_for_dndf = self._use_embedding_for_dndf = True
            dndf_input_dim = self.merged_dim
            if not self._use_embedding_for_dndf:
                dndf_input_dim -= encoding_dims["embedding"]
            if not self._use_one_hot_for_dndf:
                dndf_input_dim -= encoding_dims["one_hot"]
            self.dndf = DNDF(dndf_input_dim, self._fc_out_dim, **dndf_config)

    def _preset_config(self,
                       config: Dict[str, Any],
                       tr_data: TabularData):
        config.setdefault("default_encoding_method", ["one_hot", "embedding"])

    @staticmethod
    def _merge(split_result: SplitFeatures,
               use_embedding: bool,
               use_one_hot: bool) -> torch.Tensor:
        if use_embedding and use_one_hot:
            return split_result.merge()
        numerical = split_result.numerical
        if not use_embedding and not use_one_hot:
            return numerical
        categorical = split_result.categorical
        if not use_one_hot:
            return torch.cat([numerical, categorical["embedding"]], dim=1)
        if not use_embedding:
            return torch.cat([numerical, categorical["one_hot"]], dim=1)

    def forward(self,
                batch: tensor_dict_type,
                **kwargs) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch, return_all_encodings=True)
        # fc
        fc_net = self._merge(split_result, self._use_embedding_for_fc, self._use_one_hot_for_fc)
        fc_net = self.mlp(fc_net)
        # dndf
        dndf_net = self._merge(split_result, self._use_embedding_for_dndf, self._use_one_hot_for_dndf)
        dndf_net = self.dndf(dndf_net)
        return {"predictions": fc_net + dndf_net}


__all__ = ["TreeDNN"]
