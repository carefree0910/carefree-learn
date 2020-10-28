import torch
import logging

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import shallow_copy_dict
from cfdata.tabular import DataLoader

from ...modules.blocks import *
from ..base import ModelBase
from ..base import SplitFeatures
from ..fcnn.core import FCNN
from ...types import tensor_dict_type


@FCNN.register("tree_dnn")
class TreeDNN(FCNN):
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
        self.dndf: Optional[DNDF]
        self._dndf_config: Dict[str, Any]
        super(FCNN, self).__init__(
            pipeline_config,
            tr_loader,
            cv_loader,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        one_hot_dim = self.one_hot_dim
        embedding_dim = self.embedding_dim
        # fc
        if self.use_fcnn:
            if not self._numerical_columns:
                self._use_embedding_for_fc = True
                self._use_one_hot_for_fc = "one_hot" in self._default_encoding_method
            fc_in_dim = self.merged_dim
            if not self._use_embedding_for_fc:
                fc_in_dim -= embedding_dim * self.num_history
            if not self._use_one_hot_for_fc:
                fc_in_dim -= one_hot_dim * self.num_history
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
            self._dndf_config["is_regression"] = self.tr_data.is_reg
            self._dndf_config.setdefault("tree_proj_config", None)
            if not self._numerical_columns:
                self._use_embedding_for_dndf = True
                self._use_one_hot_for_dndf = "one_hot" in self._default_encoding_method
            dndf_input_dim = self.merged_dim
            if not self._use_embedding_for_dndf:
                dndf_input_dim -= embedding_dim * self.num_history
            if not self._use_one_hot_for_dndf:
                dndf_input_dim -= one_hot_dim * self.num_history
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
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch, batch_indices, loader_name)
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


class TreeResBlock(nn.Module):
    def __init__(self, dim: int, dndf_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if dndf_config is None:
            dndf_config = {}
        self.dim = float(dim)
        self.in_dndf = DNDF(dim, dim, **shallow_copy_dict(dndf_config))
        self.inner_dndf = DNDF(dim, dim, **shallow_copy_dict(dndf_config))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        res = self.in_dndf(net)
        res = self.dim * res - 1.0
        res = self.inner_dndf(res)
        res = self.dim * res - 1.0
        return net + res


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
        super()._init_input_config()
        dim = self._fc_in_dim
        self.res_blocks = nn.ModuleList()
        for _ in range(self._num_blocks):
            self.res_blocks.append(TreeResBlock(dim, self._dndf_config))
        self.out_dndf = DNDF(
            dim,
            self._fc_out_dim,
            **self._out_dndf_config,
        )

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    @property
    def output_probabilities(self) -> bool:
        return True

    def _init_config(self) -> None:
        super()._init_config()
        self._loss_config["input_logits"] = False
        self._num_blocks = self.config.setdefault("num_blocks", 3)
        if self._num_blocks <= 0:
            self.log_msg(  # type: ignore
                "`num_blocks` is 0 in TreeStack, it will be equivalent to TreeLinear",
                prefix=self.warning_prefix,
                verbose_level=2,
                msg_level=logging.WARNING,
            )
        self._dndf_config = self.config.setdefault("dndf_config", {})
        self._out_dndf_config = self.config.setdefault("out_dndf_config", {})

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        net = self._split_features(x_batch, batch_indices, loader_name).merge()
        if self.tr_data.is_ts:
            net = net.view(x_batch.shape[0], -1)
        for block in self.res_blocks:
            net = block(net)
        net = self.out_dndf(net)
        return {"predictions": net}


@TreeStack.register("tree_linear")
class TreeLinear(TreeStack):
    def _preset_config(self) -> None:
        super()._preset_config()
        self.config["num_blocks"] = 0


__all__ = ["TreeDNN", "TreeStack", "TreeLinear"]
