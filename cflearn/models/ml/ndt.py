import torch

import numpy as np
import torch.nn.functional as F

from typing import Dict
from typing import List
from typing import Iterator
from typing import Optional
from cftool.array import to_numpy
from cftool.array import to_torch

from .base import MLModel
from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings
from ...modules.blocks import Linear
from ...modules.blocks import Activation

try:
    from sklearn.tree import _tree
    from sklearn.tree import DecisionTreeClassifier
except:
    _tree = DecisionTreeClassifier = None


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


@MLModel.register("ndt")
class NDT(MLModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_history: int = 1,
        *,
        dt: DecisionTreeClassifier,
        encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None,
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        if DecisionTreeClassifier is None:
            raise ValueError("`scikit-learn` is needed for `NDT`")
        super().__init__(
            encoder_settings=encoder_settings,
            global_encoder_settings=global_encoder_settings,
        )
        if self.encoder is not None:
            input_dim += self.encoder.dim_increment
        input_dim *= num_history
        tree_structure = export_structure(dt)
        # dt statistics
        num_leaves = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        num_internals = num_leaves - 1
        # transform
        b = np.zeros(num_internals, dtype=np.float32)
        w1 = np.zeros([input_dim, num_internals], dtype=np.float32)
        w2 = np.zeros([num_internals, num_leaves], dtype=np.float32)
        w3 = np.zeros([num_leaves, output_dim], dtype=np.float32)
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
        w1_, w2_, w3_, b_ = map(to_torch, [w1, w2, w3, b])
        # construct planes & routes
        self.to_planes = Linear(input_dim, num_internals, init_method=None)
        self.to_routes = Linear(num_internals, num_leaves, bias=False, init_method=None)
        self.to_leaves = Linear(num_leaves, output_dim, init_method=None)
        with torch.no_grad():
            self.to_planes.linear.bias.data = b_
            self.to_planes.linear.weight.data = w1_.t()
            self.to_routes.linear.weight.data = w2_.t()
            self.to_leaves.linear.weight.data = w3_.t()
            uniform = F.log_softmax(torch.zeros(output_dim, dtype=torch.float32), dim=0)
            self.to_leaves.linear.bias.data = uniform
        # activations
        self.planes_activation = Activation.make("sign")
        self.routes_activation = Activation.make("one_hot")

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

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        if len(net.shape) > 2:
            net = net.contiguous().view(len(net), -1)
        net = self.planes_activation(self.to_planes(net))
        net = self.routes_activation(self.to_routes(net))
        net = self.to_leaves(net)
        return net


__all__ = ["NDT"]
