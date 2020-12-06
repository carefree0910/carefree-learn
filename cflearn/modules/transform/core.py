import torch

import numpy as np

from torch import Tensor
from typing import Dict
from typing import Optional
from typing import NamedTuple
from torch.nn import Module
from cftool.misc import timing_context

from ..encoders import Encoder
from ..encoders import EncodingResult
from ...misc.toolkit import LoggingMixinWithRank


class SplitFeatures(NamedTuple):
    categorical: Optional[EncodingResult]
    numerical: Optional[Tensor]

    def merge(
        self,
        use_one_hot: bool = True,
        use_embedding: bool = True,
        only_categorical: bool = False,
    ) -> Tensor:
        if use_embedding and use_one_hot:
            return self._merge_all(only_categorical)
        numerical = None if only_categorical else self.numerical
        if not use_embedding and not use_one_hot:
            if only_categorical:
                raise ValueError(
                    "`only_categorical` is set to True, "
                    "but neither `one_hot` nor `embedding` is used"
                )
            assert numerical is not None
            return numerical
        categorical = self.categorical
        if not categorical:
            if only_categorical:
                raise ValueError("categorical is not available")
            assert numerical is not None
            return numerical
        if not use_one_hot:
            embedding = categorical.embedding
            assert embedding is not None
            if numerical is None:
                return embedding
            return torch.cat([numerical, embedding], dim=1)
        one_hot = categorical.one_hot
        assert not use_embedding and one_hot is not None
        if numerical is None:
            return one_hot
        return torch.cat([numerical, one_hot], dim=1)

    def _merge_all(self, only_categorical: bool) -> Tensor:
        categorical = self.categorical
        if categorical is None:
            if only_categorical:
                raise ValueError("categorical is not available")
            assert self.numerical is not None
            return self.numerical
        merged = categorical.merged
        if only_categorical or self.numerical is None:
            return merged
        return torch.cat([self.numerical, merged], dim=1)


class Dimensions(LoggingMixinWithRank):
    def __init__(
        self,
        encoder: Optional[Encoder],
        numerical_columns_mapping: Dict[int, int],
        categorical_columns_mapping: Dict[int, int],
        num_history: int,
    ):
        self.encoder = encoder
        self._categorical_dim = 0 if encoder is None else encoder.merged_dim
        self.numerical_columns_mapping = numerical_columns_mapping
        self.categorical_columns_mapping = categorical_columns_mapping
        self._numerical_columns = sorted(numerical_columns_mapping.values())
        self.num_history = num_history

    @property
    def merged_dim(self) -> int:
        merged_dim = self._categorical_dim + self.numerical_dim
        return merged_dim * self.num_history

    @property
    def one_hot_dim(self) -> int:
        if self.encoder is None:
            return 0
        return self.encoder.one_hot_dim

    @property
    def embedding_dim(self) -> int:
        if self.encoder is None:
            return 0
        return self.encoder.embedding_dim

    @property
    def categorical_dims(self) -> Dict[int, int]:
        dims: Dict[int, int] = {}
        if self.encoder is None:
            return dims
        return self.encoder.merged_dims

    @property
    def numerical_dim(self) -> int:
        return len(self._numerical_columns)

    @property
    def has_numerical(self) -> bool:
        return self.numerical_dim > 0

    def split_features(
        self,
        x_batch: Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
        enable_timing: bool = True,
    ) -> SplitFeatures:
        if self.encoder is None:
            return SplitFeatures(None, x_batch)
        with timing_context(self, "encoding", enable=enable_timing):
            encoding_result = self.encoder(x_batch, batch_indices, loader_name)
        with timing_context(self, "fetch_numerical", enable=enable_timing):
            numerical_columns = self._numerical_columns
            if not numerical_columns:
                numerical = None
            else:
                numerical = x_batch[..., numerical_columns]
        return SplitFeatures(encoding_result, numerical)


class Transform(Module):
    def __init__(
        self,
        dimensions: Dimensions,
        *,
        one_hot: bool,
        embedding: bool,
        only_categorical: bool,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.use_one_hot = one_hot
        self.use_embedding = embedding
        self.only_categorical = only_categorical

    @property
    def out_dim(self) -> int:
        out_dim = self.dimensions.merged_dim
        if not self.use_one_hot:
            out_dim -= self.dimensions.one_hot_dim * self.dimensions.num_history
        if not self.use_embedding:
            out_dim -= self.dimensions.embedding_dim * self.dimensions.num_history
        if self.only_categorical:
            out_dim -= self.dimensions.numerical_dim
        return out_dim

    def forward(self, split: SplitFeatures) -> Tensor:
        return split.merge(self.use_one_hot, self.use_embedding, self.only_categorical)

    def extra_repr(self) -> str:
        one_hot_str = f"(use_one_hot): {self.use_one_hot}"
        embedding_str = f"(use_embedding): {self.use_embedding}"
        only_str = f"(only_categorical): {self.only_categorical}"
        return f"{one_hot_str}\n{embedding_str}\n{only_str}"


__all__ = ["SplitFeatures", "Dimensions", "Transform"]
