import numpy as np
import torch.nn as nn

from sklearn.tree import DecisionTreeClassifier

from ..base import ModelBase
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch


@ModelBase.register("ndt")
@ModelBase.register_pipe("ndt")
class NDT(ModelBase):
    @property
    def head(self) -> nn.Module:
        return self.pipes["ndt"].head

    @property
    def hyperplane_weights(self) -> np.ndarray:
        return to_numpy(self.head.to_planes.linear.weight)

    @property
    def hyperplane_thresholds(self) -> np.ndarray:
        return to_numpy(-self.head.to_planes.linear.bias)

    @property
    def route_weights(self) -> np.ndarray:
        return to_numpy(self.head.to_routes.linear.weight)

    @property
    def class_log_distributions(self) -> np.ndarray:
        return to_numpy(self.head.to_leaves.linear.weight)

    @property
    def class_log_prior(self) -> np.ndarray:
        return to_numpy(self.head.to_leaves.linear.bias)

    @property
    def class_prior(self) -> np.ndarray:
        return np.exp(self.class_log_prior)

    def define_pipe_configs(self) -> None:
        # prepare
        x, y = self.tr_data.processed.xy
        y_ravel, num_classes = y.ravel(), self.tr_data.num_classes
        # decision tree
        dt_config = self.config.setdefault("dt_config", {})
        dt_config.setdefault("max_depth", 10)
        msg = "fitting decision tree"
        self.log_msg(msg, self.info_prefix, verbose_level=2)  # type: ignore
        split = self._split_features(to_torch(x), np.arange(len(x)), "tr")
        x_merge = split.merge().cpu().numpy()
        self.dt = DecisionTreeClassifier(**dt_config, random_state=142857)
        self.dt.fit(x_merge, y_ravel, sample_weight=self.tr_weights)
        # activations
        activation_configs = self.config.setdefault("activation_configs", {})
        default_activations = {"planes": "sign", "routes": "multiplied_softmax"}
        activations = self.config.setdefault("activations", default_activations)
        self.define_head_config(
            "ndt",
            {
                "dt": self.dt,
                "num_classes": num_classes,
                "merged_dim": self.merged_dim,
                "activations": activations,
                "activation_configs": activation_configs,
            },
        )

    def _preset_config(self) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")


__all__ = ["NDT"]
