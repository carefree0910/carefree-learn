import numpy as np
import torch.nn as nn

from .base import ModelBase
from ..misc.toolkit import to_numpy
from ..modules.blocks import Linear


@ModelBase.register("ndt")
@ModelBase.register_pipe("ndt")
class NDT(ModelBase):
    def _preset_config(self) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")

    @property
    def head(self) -> nn.Module:
        return self.heads["ndt"]

    @property
    def to_planes(self) -> Linear:
        return self.head.to_planes  # type: ignore

    @property
    def to_routes(self) -> Linear:
        return self.head.to_routes  # type: ignore

    @property
    def to_leaves(self) -> Linear:
        return self.head.to_leaves  # type: ignore

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


__all__ = ["NDT"]
