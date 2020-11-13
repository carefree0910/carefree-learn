import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from functools import partial
from cfdata.types import np_int_type
from torch.nn.functional import linear
from torch.nn.functional import softmax
from torch.nn.functional import log_softmax
from cfml.models.naive_bayes import MultinomialNB

from ...misc.toolkit import *
from ...modules.blocks import *
from ..base import ModelBase
from ...types import tensor_dict_type


@ModelBase.register("nnb")
class NNB(ModelBase):
    def define_transforms(self) -> None:
        self.add_transform("numerical", False, False, False)
        self.add_transform("categorical", True, False, True)

    def define_extractors(self) -> None:
        self.add_extractor("numerical")
        self.add_extractor("categorical")

    def define_heads(self) -> None:
        # prepare
        x, y = self.tr_data.processed.xy
        y_ravel, num_classes = y.ravel(), self.tr_data.num_classes
        split = self._split_features(to_torch(x), np.arange(len(x)), "tr")
        # numerical
        num_numerical = len(self._numerical_columns)
        if num_numerical == 0:
            self.mu = self.std = None
        else:
            if not self.pretrain:
                self.mu = nn.Parameter(torch.zeros(num_classes, num_numerical))
                self.std = nn.Parameter(torch.ones(num_classes, num_numerical))
            else:
                numerical = self.extract("numerical", split)
                assert isinstance(numerical, torch.Tensor)
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
            normal = torch.distributions.Normal(self.mu, self.std)
            normal_head = Lambda(
                lambda net: normal.log_prob(net[..., None, :]).sum(2),
                "normal_head",
            )
            self.add_head("numerical", normal_head)
        # categorical
        one_hot_dim = self.one_hot_dim
        if one_hot_dim == 0:
            self.mnb = None
            y_bincount = np.bincount(y_ravel).astype(np.float32)
            self.log_prior = nn.Parameter(torch.from_numpy(np.log(y_bincount / len(x))))
        else:
            self.mnb = Linear(one_hot_dim, num_classes, init_method=None)
            categorical = self.extract("categorical", split)
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
            mnb_head = Lambda(
                lambda net: linear(
                    net,
                    self.log_posterior(),
                    self.class_log_prior(),
                ),
                "mnb_head",
            )
            self.add_head("categorical", mnb_head)

    def _preset_config(self) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")

    def _init_config(self) -> None:
        super()._init_config()
        self.pretrain = self.config.setdefault("pretrain", True)

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

    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split = self._split_features(x_batch, batch_indices, loader_name)
        # numerical
        if self.mu is None:
            numerical_log_prob = None
        else:
            numerical_log_prob = self.execute("numerical", split)
        # categorical
        if self.mnb is None:
            categorical_log_prob = None
            numerical_log_prob = numerical_log_prob + self.class_log_prior()
        else:
            categorical_log_prob = self.execute("categorical", split)
        if numerical_log_prob is None:
            predictions = categorical_log_prob
        elif categorical_log_prob is None:
            predictions = numerical_log_prob
        else:
            predictions = numerical_log_prob + categorical_log_prob
        return {"predictions": predictions}

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


__all__ = ["NNB"]
