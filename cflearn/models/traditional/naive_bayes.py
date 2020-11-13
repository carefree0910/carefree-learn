import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Optional
from functools import partial
from cftool.misc import shallow_copy_dict
from torch.nn.functional import softmax

from ...misc.toolkit import *
from ..base import ModelBase


@ModelBase.register("nnb")
@ModelBase.register_pipe("nnb_mnb")
@ModelBase.register_pipe("nnb_normal")
class NNB(ModelBase):
    def _preset_config(self) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")

    def _init_config(self) -> None:
        super()._init_config()
        self.pretrain = self.config.setdefault("pretrain", True)

    @property
    def mnb(self) -> nn.Module:
        return self.pipes["nnb_mnb"].head

    @property
    def normal(self) -> nn.Module:
        return self.pipes["nnb_normal"].head

    @property
    def class_log_prior(self) -> Callable:
        return self.mnb.class_log_prior

    @property
    def log_posterior(self) -> Callable:
        return self.mnb.log_posterior

    @property
    def pdf(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if self.normal is None:
            return None

        def _pdf(arr: np.ndarray) -> np.ndarray:
            if self.mu is None:
                raise ValueError("`mu` is not trained")
            assert self.normal is not None
            tensor = to_torch(arr).to(self.mu.device)
            pdf = torch.exp(self.normal.log_prob(tensor[..., None, :]))
            return to_numpy(pdf)

        return _pdf

    @property
    def class_prior(self) -> np.ndarray:
        return np.exp(self.class_log_prior(numpy=True))

    @property
    def posteriors(self) -> Optional[Tuple[np.ndarray, ...]]:
        if self.mnb is None:
            return None
        with torch.no_grad():
            posteriors = tuple(
                map(
                    partial(softmax, dim=1),
                    self.log_posterior(return_groups=True),
                )
            )
        return tuple(map(to_numpy, posteriors))

    def define_pipe_configs(self) -> None:
        # prepare
        x, y = self.tr_data.processed.xy
        y_ravel, num_classes = y.ravel(), self.tr_data.num_classes
        split = self._split_features(to_torch(x), np.arange(len(x)), "tr")
        if not self._numerical_columns:
            numerical = None
        else:
            numerical = split.numerical
        if self.one_hot_dim == 0:
            categorical = None
        else:
            categorical = split.categorical.one_hot
        common_config = {"y_ravel": y_ravel, "num_classes": num_classes}
        # mnb
        mnb_config = shallow_copy_dict(common_config)
        mnb_config.update({"one_hot_dim": self.one_hot_dim, "categorical": categorical})
        self.define_head_config("nnb_mnb", mnb_config)
        self.define_transform_config(
            "nnb_mnb",
            {"one_hot": True, "embedding": False, "only_categorical": True},
        )
        # normal
        normal_config = shallow_copy_dict(common_config)
        normal_config.update({"numerical": numerical, "pretrain": self.pretrain})
        self.define_head_config("nnb_normal", normal_config)
        self.define_transform_config(
            "nnb_normal",
            {"one_hot": False, "embedding": False, "only_categorical": False},
        )

    def merge_outputs(
        self,
        outputs: Dict[str, Tensor],
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        # numerical
        if self.normal.mu is None:
            numerical_log_prob = None
        else:
            numerical_log_prob = outputs["nnb_normal"]
        # categorical
        if self.mnb.mnb is None:
            categorical_log_prob = None
            numerical_log_prob = numerical_log_prob + self.class_log_prior()
        else:
            categorical_log_prob = outputs["nnb_mnb"]
        if numerical_log_prob is None:
            predictions = categorical_log_prob
        elif categorical_log_prob is None:
            predictions = numerical_log_prob
        else:
            predictions = numerical_log_prob + categorical_log_prob
        return {"predictions": predictions}


__all__ = ["NNB"]
