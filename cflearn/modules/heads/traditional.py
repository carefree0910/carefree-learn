import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Union
from typing import Optional
from cfdata.types import np_int_type
from torch.nn.functional import linear
from torch.nn.functional import log_softmax
from cfml.models.naive_bayes import MultinomialNB

from .base import HeadBase
from ...misc.toolkit import to_numpy
from ...misc.toolkit import to_torch
from ...misc.toolkit import Lambda
from ...modules.blocks import Linear


@HeadBase.register("nnb_mnb")
class NNBMNBHead(HeadBase):
    def __init__(
        self,
        y_ravel: np.ndarray,
        num_classes: int,
        one_hot_dim: int,
        categorical: Optional[torch.Tensor],
    ):
        super().__init__()
        if categorical is None:
            self.mnb = None
            y_bincount = np.bincount(y_ravel).astype(np.float32)
            log_prior = torch.from_numpy(np.log(y_bincount / len(y_ravel)))
            self.log_prior = nn.Parameter(log_prior)
        else:
            self.mnb = Linear(one_hot_dim, num_classes, init_method=None)
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
        if self.mnb is None:
            return net
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
        y_ravel: np.ndarray,
        num_classes: int,
        pretrain: bool,
        numerical: Optional[torch.Tensor],
    ):
        super().__init__()
        if numerical is None:
            self.mu = self.std = self.normal = None
        else:
            num_numerical = numerical.shape[1]
            if not pretrain:
                self.mu = nn.Parameter(torch.zeros(num_classes, num_numerical))
                self.std = nn.Parameter(torch.ones(num_classes, num_numerical))
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
        if self.normal is None:
            return net
        return self.normal.log_prob(net[..., None, :]).sum(2)


__all__ = [
    "NDTHead",
    "NNBMNBHead",
    "NNBNormalHead",
]
