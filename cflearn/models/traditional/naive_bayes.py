import torch

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
from functools import partial
from cfdata.types import np_int_type
from cfdata.tabular import TabularData
from cfml.models.naive_bayes import MultinomialNB

from ..base import ModelBase
from ...misc.toolkit import *
from ...modules.blocks import *


@ModelBase.register("nnb")
class NNB(ModelBase):
    def __init__(
        self,
        config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super().__init__(config, tr_data, device)
        # prepare
        x, y = tr_data.processed.xy
        y_ravel, num_classes = y.ravel(), tr_data.num_classes
        x_tensor = torch.from_numpy(x).to(device)
        split_result = self._split_features(x_tensor)
        # numerical
        num_numerical = len(self._numerical_columns)
        if num_numerical == 0:
            self.mu = self.std = self.normal = None
        else:
            if not self.pretrain:
                self.mu = nn.Parameter(torch.zeros(num_classes, num_numerical))
                self.std = nn.Parameter(torch.ones(num_classes, num_numerical))
            else:
                numerical = split_result.numerical
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
            self.normal = torch.distributions.Normal(self.mu, self.std)
        # categorical
        num_categorical_dim = self.encoding_dims.get("one_hot", 0)
        if num_categorical_dim == 0:
            self.mnb = None
            y_bincount = np.bincount(y_ravel).astype(np.float32)
            self.log_prior = nn.Parameter(torch.from_numpy(np.log(y_bincount / len(x))))
        else:
            self.mnb = Linear(num_categorical_dim, num_classes, init_method=None)
            categorical = split_result.categorical
            assert isinstance(categorical, torch.Tensor)
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

    @property
    def input_sample(self) -> tensor_dict_type:
        return super().input_sample

    def _preset_config(self, tr_data: TabularData) -> None:
        self.config.setdefault("default_encoding_method", "one_hot")

    def _init_config(self, tr_data: TabularData) -> None:
        super()._init_config(tr_data)
        self.pretrain = self.config.setdefault("pretrain", True)

    @property
    def pdf(self) -> Union[Callable[[np.ndarray], np.ndarray], None]:
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
    def posteriors(self) -> Union[Tuple[np.ndarray, ...], None]:
        if self.mnb is None:
            return None
        with torch.no_grad():
            posteriors = tuple(
                map(
                    partial(nn.functional.softmax, dim=1),
                    self.log_posterior(return_groups=True),
                )
            )
        return tuple(map(to_numpy, posteriors))

    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split_result = self._split_features(x_batch)
        # log prior
        log_prior = self.class_log_prior()
        # numerical
        if self.normal is None:
            numerical_log_prob = None
        else:
            numerical = split_result.numerical
            assert isinstance(numerical, torch.Tensor)
            numerical_log_prob = self.normal.log_prob(numerical[..., None, :]).sum(2)
        # categorical
        if self.mnb is None:
            categorical_log_prob = None
            numerical_log_prob = numerical_log_prob + log_prior
        else:
            categorical = split_result.categorical
            log_posterior = self.log_posterior()
            assert isinstance(categorical, torch.Tensor)
            categorical_log_prob = nn.functional.linear(
                categorical,
                log_posterior,
                log_prior,
            )
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
        rs = nn.functional.log_softmax(log_prior, dim=0)
        if not numpy:
            return rs
        return to_numpy(rs)

    def log_posterior(
        self,
        *,
        numpy: bool = False,
        return_groups: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        mnb = self.mnb
        if mnb is None:
            raise ValueError("`mnb` is not trained")
        rs = nn.functional.log_softmax(mnb.linear.weight, dim=1)
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
            lambda tensor: nn.functional.log_softmax(tensor, dim=1),
            torch.split(rs, num_categorical_list, dim=1),
        )
        if not numpy:
            return tuple(grouped_weights)
        return tuple(map(to_numpy, grouped_weights))


__all__ = ["NNB"]
