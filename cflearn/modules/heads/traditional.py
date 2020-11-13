import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Iterator
from typing import Optional
from cfdata.types import np_int_type
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from torch.nn.functional import linear
from torch.nn.functional import log_softmax
from cfml.models.naive_bayes import MultinomialNB

from .base import HeadBase
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch
from ...misc.toolkit import Activations
from ...modules.blocks import Linear


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


@HeadBase.register("ndt")
class NDTHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        dt: DecisionTreeClassifier,
        num_classes: int,
        activations: Dict[str, str],
        activation_configs: Dict[str, Any],
    ):
        super().__init__()
        tree_structure = export_structure(dt)
        # dt statistics
        num_leaves = sum([1 if pair[1] == -1 else 0 for pair in tree_structure])
        num_internals = num_leaves - 1
        msg = f"internals : {num_internals} ; leaves : {num_leaves}"
        self.log_msg(msg, self.info_prefix, verbose_level=2)  # type: ignore
        # transform
        b = np.zeros(num_internals, dtype=np.float32)
        w1 = np.zeros([in_dim, num_internals], dtype=np.float32)
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
        self.to_planes = Linear(in_dim, num_internals, init_method=None)
        self.to_routes = Linear(num_internals, num_leaves, bias=False, init_method=None)
        self.to_leaves = Linear(num_leaves, num_classes, init_method=None)
        with torch.no_grad():
            self.to_planes.linear.bias.data = b
            self.to_planes.linear.weight.data = w1.t()
            self.to_routes.linear.weight.data = w2.t()
            self.to_leaves.linear.weight.data = w3.t()
            uniform = log_softmax(torch.zeros(num_classes, dtype=torch.float32), dim=0)
            self.to_leaves.linear.bias.data = uniform
        # activations
        m_tanh_cfg = activation_configs.setdefault("multiplied_tanh", {})
        m_sm_cfg = activation_configs.setdefault("multiplied_softmax", {})
        m_tanh_cfg.setdefault("ratio", 10.0)
        m_sm_cfg.setdefault("ratio", 10.0)
        activations_ins = Activations(activation_configs)
        self.planes_activation = activations_ins.module(activations.get("planes"))
        self.routes_activation = activations_ins.module(activations.get("routes"))

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        planes = self.planes_activation(self.to_planes(net))
        routes = self.routes_activation(self.to_routes(planes))
        return self.to_leaves(routes)


@HeadBase.register("nnb_mnb")
class NNBMNBHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        y_ravel: np.ndarray,
        num_classes: int,
        categorical: Optional[torch.Tensor],
    ):
        super().__init__()
        if categorical is None:
            self.mnb = None
            y_bincount = np.bincount(y_ravel).astype(np.float32)
            log_prior = torch.from_numpy(np.log(y_bincount / len(y_ravel)))
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.mnb = Linear(in_dim, num_classes, init_method=None)
            x_mnb = categorical.cpu().numpy()
            y_mnb = y_ravel.astype(np_int_type)
            mnb = MultinomialNB().fit(x_mnb, y_mnb)
            with torch.no_grad():
                # class log prior
                class_log_prior = mnb.class_log_prior[0]
                self.mnb.linear.bias.data = to_torch(class_log_prior)
                assert np.allclose(
                    class_log_prior, self.class_log_prior(numpy=True), atol=1e-6
                )
                # log posterior
                self.mnb.linear.weight.data = to_torch(mnb.feature_log_prob)
                assert np.allclose(
                    mnb.feature_log_prob, self.log_posterior(numpy=True), atol=1e-6
                )

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return linear(net, self.log_posterior(), self.class_log_prior())

    def class_log_prior(
        self,
        *,
        numpy: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        log_prior = self.log_prior if self.mnb is None else self.mnb.linear.bias
        rs = log_softmax(log_prior, dim=0)
        if not numpy:
            return rs
        return to_numpy(rs)

    def log_posterior(
        self,
        *,
        numpy: bool = False,
        return_groups: bool = False,
    ) -> Any:
        mnb = self.mnb
        if mnb is None:
            raise ValueError("`mnb` is not trained")
        rs = log_softmax(mnb.linear.weight, dim=1)
        if not return_groups:
            if not numpy:
                return rs
            return to_numpy(rs)
        categorical_dims = self.categorical_dims
        sorted_categorical_indices = sorted(categorical_dims)
        num_categorical_list = [
            categorical_dims[idx] for idx in sorted_categorical_indices
        ]
        grouped_weights = map(
            lambda tensor: log_softmax(tensor, dim=1),
            torch.split(rs, num_categorical_list, dim=1),
        )
        if not numpy:
            return tuple(grouped_weights)
        return tuple(map(to_numpy, grouped_weights))


@HeadBase.register("nnb_normal")
class NNBNormalHead(HeadBase):
    def __init__(
        self,
        in_dim: int,
        y_ravel: np.ndarray,
        num_classes: int,
        pretrain: bool,
        numerical: Optional[torch.Tensor],
    ):
        super().__init__()
        if numerical is None:
            self.mu = self.std = self.normal = None
        else:
            if not pretrain:
                self.mu = nn.Parameter(torch.zeros(num_classes, in_dim))
                self.std = nn.Parameter(torch.ones(num_classes, in_dim))
            else:
                x_numerical = numerical.cpu().numpy()
                mu_list, std_list = [], []
                for k in range(num_classes):
                    local_samples = x_numerical[y_ravel == k]
                    mu_list.append(local_samples.mean(0))
                    std_list.append(local_samples.std(0))
                self.mu, self.std = map(
                    lambda lst: nn.Parameter(torch.from_numpy(np.vstack(lst))),
                    [mu_list, std_list],
                )
            self.normal = torch.distributions.Normal(self.mu, self.std)

    def forward(self, net: torch.Tensor) -> torch.Tensor:
        return self.normal.log_prob(net[..., None, :]).sum(2)


__all__ = [
    "NDTHead",
    "NNBMNBHead",
    "NNBNormalHead",
]
