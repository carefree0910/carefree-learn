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
from torch.nn.functional import softmax

from ..misc.toolkit import *
from .base import ModelBase


@ModelBase.register("nnb")
@ModelBase.register_pipe(
    "nnb_mnb",
    transform="one_hot_only",
    head_meta_scope="nnb_meta",
)
@ModelBase.register_pipe(
    "nnb_normal",
    transform="numerical",
    head_meta_scope="nnb_meta",
)
class NNB(ModelBase):
    @property
    def mnb(self) -> nn.Module:
        return self.heads["nnb_mnb"][0]

    @property
    def normal(self) -> nn.Module:
        return self.heads["nnb_normal"][0]

    @property
    def class_log_prior(self) -> Callable:
        return self.mnb.class_log_prior  # type: ignore

    @property
    def log_posterior(self) -> Callable:
        return self.mnb.log_posterior  # type: ignore

    @property
    def pdf(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if self.normal is None:
            return None

        def _pdf(arr: np.ndarray) -> np.ndarray:
            if self.mu is None:
                raise ValueError("`mu` is not trained")
            assert self.normal is not None
            tensor = to_torch(arr).to(self.mu.device)  # type: ignore
            pdf = torch.exp(self.normal.log_prob(tensor[..., None, :]))  # type: ignore
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
