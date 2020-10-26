import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Iterator
from typing import Optional
from cfdata.tabular import DataLoader
from cfdata.tabular import TabularData
from sklearn.tree import _tree, DecisionTreeClassifier

from ...misc.toolkit import *
from ...modules.blocks import *
from ..base import ModelBase
from ...types import tensor_dict_type


def export_structure(tree: DecisionTreeClassifier) -> tuple:
    tree = tree.tree_

    def recurse(node: int, depth: int) -> Iterator:
        feature_dim = tree.feature[node]
        if feature_dim == _tree.TREE_UNDEFINED:
            yield depth, -1, tree.value[node]
        else:
            threshold = tree.threshold[node]
            yield depth, feature_dim, threshold
            yield from recurse(tree.children_left[node], depth + 1)
            yield depth, feature_dim, threshold
            yield from recurse(tree.children_right[node], depth + 1)

    return tuple(recurse(0, 0))


@ModelBase.register("ndt")
class NDT(ModelBase):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_data: TabularData,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__(
            pipeline_config,
            tr_loader,
            cv_data,
            tr_weights,
            cv_weights,
            device,
            use_tqdm=use_tqdm,
        )
        # prepare
        x, y = self.tr_data.processed.xy
        y_ravel, num_classes = y.ravel(), self.tr_data.num_classes
        x_tensor = torch.from_numpy(x).to(device)
        split_result = self._split_features(x_tensor, np.arange(len(x_tensor)))
        # decision tree
        msg = "fitting decision tree"
        self.log_msg(msg, self.info_prefix, verbose_level=2)  # type: ignore
        x_merge = split_result.merge().cpu().numpy()
        self.dt = DecisionTreeClassifier(**self.dt_config, random_state=142857)
        self.dt.fit(x_merge, y_ravel, sample_weight=tr_weights)
        tree_structure = export_structure(self.dt)
        # dt statistics
        num_leaves = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        num_internals = num_leaves - 1
        msg = f"internals : {num_internals} ; leaves : {num_leaves}"
        self.log_msg(msg, self.info_prefix, verbose_level=2)  # type: ignore
        # transform
        b = np.zeros(num_internals, dtype=np.float32)
        w1 = np.zeros([self.merged_dim, num_internals], dtype=np.float32)
        w2 = np.zeros([num_internals, num_leaves], dtype=np.float32)
        w3 = np.zeros([num_leaves, num_classes], dtype=np.float32)
        node_list: List[int] = []
        node_sign_list: List[int] = []
        node_id_cursor = leaf_id_cursor = 0
        for depth, feat_dim, rs in tree_structure:
            if feat_dim != -1:
                if depth == len(node_list):
                    node_sign_list.append(-1)
                    node_list.append(node_id_cursor)
                    w1[feat_dim, node_id_cursor] = 1
                    b[node_id_cursor] = -rs
                    node_id_cursor += 1
                else:
                    node_list = node_list[: depth + 1]
                    node_sign_list = node_sign_list[:depth] + [1]
            else:
                for node_id, node_sign in zip(node_list, node_sign_list):
                    w2[node_id, leaf_id_cursor] = node_sign / len(node_list)
                w3[leaf_id_cursor] = rs / np.sum(rs)
                leaf_id_cursor += 1
        w1, w2, w3, b = map(torch.from_numpy, [w1, w2, w3, b])
        # construct planes & routes
        self.to_planes = Linear(self.merged_dim, num_internals, init_method=None)
        self.to_routes = Linear(num_internals, num_leaves, bias=False, init_method=None)
        self.to_leaves = Linear(num_leaves, num_classes, init_method=None)
        with torch.no_grad():
            self.to_planes.linear.bias.data = b
            self.to_planes.linear.weight.data = w1.t()
            self.to_routes.linear.weight.data = w2.t()
            self.to_leaves.linear.weight.data = w3.t()
            uniform = nn.functional.log_softmax(
                torch.zeros(num_classes, dtype=torch.float32), dim=0
            )
            self.to_leaves.linear.bias.data = uniform

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    @property
    def hyperplane_weights(self) -> np.ndarray:
        return to_numpy(self.to_planes.linear.weight)

    @property
    def hyperplane_thresholds(self) -> np.ndarray:
        return to_numpy(-self.to_planes.linear.bias)

    @property
    def route_weights(self) -> np.ndarray:
        return to_numpy(self.to_routes.linear.weight)

    @property
    def class_log_distributions(self) -> np.ndarray:
        return to_numpy(self.to_leaves.linear.weight)

    @property
    def class_log_prior(self) -> np.ndarray:
        return to_numpy(self.to_leaves.linear.bias)

    @property
    def class_prior(self) -> np.ndarray:
        return np.exp(self.class_log_prior)

    def _preset_config(self) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")

    def _init_config(self) -> None:
        super()._init_config()
        self.dt_config = self.config.setdefault("dt_config", {})
        self.dt_config.setdefault("max_depth", 10)
        activation_configs = self.config.setdefault("activation_configs", {})
        activation_configs.setdefault("multiplied_tanh", {}).setdefault("ratio", 10.0)
        activation_configs.setdefault("multiplied_softmax", {}).setdefault(
            "ratio", 10.0
        )
        default_activations = {"planes": "sign", "routes": "multiplied_softmax"}
        activations = self.config.setdefault("activations", default_activations)
        activations_ins = Activations(activation_configs)
        self.planes_activation = activations_ins.module(activations.get("planes"))
        self.routes_activation = activations_ins.module(activations.get("routes"))
        self._init_with_dt = self.config.setdefault("")

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        merged = self._split_features(x_batch, batch_indices).merge()
        planes = self.planes_activation(self.to_planes(merged))
        routes = self.routes_activation(self.to_routes(planes))
        leaves = self.to_leaves(routes)
        return {"predictions": leaves}


__all__ = ["NDT"]
