import math
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import *
from abc import ABCMeta
from collections import defaultdict
from cftool.misc import LoggingMixin
from cfdata.tabular import DataLoader
from cfdata.tabular.misc import np_int_type

from ..misc.toolkit import to_torch
from ..misc.toolkit import Lambda
from ..misc.toolkit import Initializer


class EncodingResult(NamedTuple):
    one_hot: Optional[torch.Tensor]
    embedding: Optional[torch.Tensor]

    @property
    def merged(self) -> Optional[torch.Tensor]:
        if self.one_hot is None and self.embedding is None:
            return None
        if self.one_hot is None:
            return self.embedding
        if self.embedding is None:
            return self.one_hot
        return torch.cat([self.one_hot, self.embedding], dim=1)


class OneHot(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        one_hot_fn = lambda column: nn.functional.one_hot(column, dim)
        self.core = Lambda(one_hot_fn, f"one_hot_{dim}")
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.core(tensor).to(torch.float32)


class Embedding(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_method: Optional[str],
        init_config: Dict[str, Any],
    ):
        super().__init__()
        weights = torch.empty(in_dim, out_dim)
        if init_method is None:
            nn.init.normal_(weights)
        else:
            Initializer(init_config).initialize(weights, init_method)
        self.weights = nn.Parameter(weights)
        embedding_fn = lambda column: nn.functional.embedding(column, self.weights)
        self.core = Lambda(embedding_fn, f"embedding: {in_dim} -> {out_dim}")
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.core(tensor)


# TODO : pass in Dict[str, DataLoader] instead of DataLoader
class Encoder(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        input_dims: List[int],
        methods_list: List[Union[str, List[str]]],
        configs: List[Dict[str, Any]],
        categorical_columns: List[int],
        loader: DataLoader,
    ):
        super().__init__()
        if loader.enabled_sampling:
            raise ValueError("`loader` should not enable sampling in `Encoder`")
        if loader.sampler.shuffle:
            raise ValueError("`loader` should not be shuffled in `Encoder`")
        self.merged_dim = 0
        self.one_hot_dim = 0
        self.embedding_dim = 0
        self.tgt_columns = np.array(sorted(categorical_columns), np_int_type)
        self.merged_dims = defaultdict(int)
        self.embeddings = nn.ModuleDict()
        self.one_hot_encoders = nn.ModuleDict()
        self._one_hot_indices = []
        self._embed_indices = []
        for i, (in_dim, methods, config) in enumerate(
            zip(input_dims, methods_list, configs)
        ):
            if isinstance(methods, str):
                methods = [methods]
            self._register(i, in_dim, methods, config)
        self.one_hot_columns = self.tgt_columns[self._one_hot_indices]
        self.embedding_columns = self.tgt_columns[self._embed_indices]
        self._compile(loader)

    @property
    def use_one_hot(self) -> bool:
        return len(self._one_hot_indices) > 0

    @property
    def use_embedding(self) -> bool:
        return len(self._embed_indices) > 0

    # TODO : pass in `name`, which indicates the name of DataLoader
    def forward(
        self,
        x_batch: torch.Tensor,
        batch_indices: Optional[np.ndarray],
    ) -> EncodingResult:
        # one hot
        if not self.use_one_hot:
            one_hot = None
        else:
            if batch_indices is not None:
                one_hot = self.one_hot_cache[batch_indices]
            else:
                one_hot_columns = x_batch[..., self.one_hot_columns]
                one_hot_encodings = self._one_hot(one_hot_columns)
                one_hot = torch.cat(one_hot_encodings, dim=1)
        # embedding
        if not self._embed_indices:
            embedding = None
        else:
            indices = x_batch[..., self.tgt_columns[self._embed_indices]]
            # if batch_indices is not None:
            #     indices = self.indices_cache[batch_indices]
            # else:
            #     indices = x_batch[..., self.embedding_columns].to(torch.long)
            embedding_encodings = self._embedding(indices.to(torch.long))
            embedding = torch.cat(embedding_encodings, dim=1)
        return EncodingResult(one_hot, embedding)

    def _register(
        self,
        i: int,
        in_dim: int,
        methods: List[str],
        config: Dict[str, Any],
    ) -> None:
        for method in methods:
            attr = getattr(self, f"_register_{method}", None)
            if attr is None:
                msg = f"encoding method '{method}' is not implemented"
                raise NotImplementedError(msg)
            attr(i, in_dim, config)

    def _register_one_hot(self, i: int, in_dim: int, config: Dict[str, Any]) -> None:
        self.one_hot_encoders[str(i)] = OneHot(in_dim)
        self._one_hot_indices.append(i)
        self.merged_dims[i] += in_dim
        self.one_hot_dim += in_dim
        self.merged_dim += in_dim

    @staticmethod
    def _get_embed_key(num: int) -> str:
        return f"embedding_weight_{num}"

    def _register_embedding(self, i: int, in_dim: int, config: Dict[str, Any]) -> None:
        self._embed_indices.append(i)
        init_method = config.setdefault("init_method", "truncated_normal")
        init_config = config.setdefault("init_config", {"mean": 0.0, "std": 0.02})
        embedding_dim = config.setdefault("embedding_dim", "auto")
        if isinstance(embedding_dim, int):
            out_dim = embedding_dim
        elif embedding_dim == "log":
            out_dim = math.ceil(math.log2(in_dim))
        elif embedding_dim == "sqrt":
            out_dim = math.ceil(math.sqrt(in_dim))
        elif embedding_dim == "auto":
            out_dim = min(in_dim, max(4, min(8, math.ceil(math.log2(in_dim)))))
        else:
            raise ValueError(f"embedding dim '{embedding_dim}' is not defined")
        self.merged_dims[i] += out_dim
        self.embedding_dim += out_dim
        self.merged_dim += out_dim
        self.embeddings[str(i)] = Embedding(in_dim, out_dim, init_method, init_config)

    @staticmethod
    def _get_dim_sum(encodings: List[torch.Tensor], indices: List[int]) -> int:
        if not indices:
            return 0
        return sum([encodings[i].shape[1] for i in indices])

    # TODO : apply this method to `_one_hot` & `_embedding`
    def _oob_imputation(self, flat_features: torch.Tensor, num_values: int) -> None:
        oob_mask = flat_features >= num_values
        if torch.any(oob_mask):
            self.log_msg(  # type: ignore
                f"out of bound occurred in categorical column {self.idx}, "
                f"ratio : {torch.mean(oob_mask.to(torch.float)).item():8.6f}",
                prefix=self.warning_prefix,
                verbose_level=5,
                msg_level=logging.WARNING,
            )
            # TODO : currently pytorch does not support onnx with bool masks
            #        in the future this line should be un-indented
            flat_features[oob_mask] = 0

    def _one_hot(self, one_hot_columns: torch.Tensor) -> List[torch.Tensor]:
        one_hot_encodings = []
        one_hot_columns = one_hot_columns.to(torch.long)
        for i, flat_feature in enumerate(one_hot_columns.t()):
            encoder = self.one_hot_encoders[str(i)]
            one_hot_encodings.append(encoder(flat_feature))
        return one_hot_encodings

    def _embedding(self, indices_columns: torch.Tensor) -> List[torch.Tensor]:
        embedding_encodings = []
        for i, flat_indices in enumerate(indices_columns.t()):
            embedding = self.embeddings[str(i)]
            embedding_encodings.append(embedding(flat_indices))
        return embedding_encodings

    def _compile(self, loader: DataLoader) -> None:
        categorical_features = []
        return_indices = loader.return_indices
        for a, b in loader:
            if return_indices:
                x_batch, y_batch = a
            else:
                x_batch, y_batch = a, b
            categorical_features.append(x_batch[..., self.tgt_columns])
        tensor = to_torch(np.vstack(categorical_features))
        # compile one hot
        if self.use_one_hot:
            one_hot_encodings = self._one_hot(tensor[..., self._one_hot_indices])
            one_hot_cache = torch.cat(one_hot_encodings, dim=1)
            self.register_buffer("one_hot_cache", one_hot_cache)
        # compile embedding
        if self.use_embedding:
            embedding_indices = tensor[..., self._embed_indices].to(torch.long)
            self.register_buffer("indices_cache", embedding_indices)


__all__ = ["Encoder", "EncodingResult"]
