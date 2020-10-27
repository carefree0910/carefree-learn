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
from ..modules.auxiliary import Dropout


class EncodingResult(NamedTuple):
    one_hot: Optional[torch.Tensor]
    embedding: Optional[torch.Tensor]

    @property
    def merged(self) -> torch.Tensor:
        if self.one_hot is None and self.embedding is None:
            raise ValueError("no data is provided in `EncodingResult`")
        if self.one_hot is None:
            assert self.embedding is not None
            return self.embedding
        if self.embedding is None:
            assert self.one_hot is not None
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


class Encoder(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        input_dims: List[int],
        methods_list: List[Union[str, List[str]]],
        configs: List[Dict[str, Any]],
        categorical_columns: List[int],
        loaders: Dict[str, DataLoader],
        embedding_dropout: float,
    ):
        super().__init__()
        for loader in loaders.values():
            if loader.enabled_sampling:
                raise ValueError("`loader` should not enable sampling in `Encoder`")
            if loader.sampler.shuffle:
                raise ValueError("`loader` should not be shuffled in `Encoder`")
        self.merged_dim = 0
        self.one_hot_dim = 0
        self.embedding_dim = 0
        bounds = torch.tensor(input_dims, dtype=torch.float32) - 1.0
        self.register_buffer("bounds", bounds)
        self.tgt_columns = np.array(sorted(categorical_columns), np_int_type)
        self.merged_dims: Dict[int, int] = defaultdict(int)
        self.embeddings = nn.ModuleList()
        self.one_hot_encoders = nn.ModuleList()
        self._one_hot_indices: List[int] = []
        self._embed_indices: List[int] = []
        for i, (in_dim, methods, config) in enumerate(
            zip(input_dims, methods_list, configs)
        ):
            if isinstance(methods, str):
                methods = [methods]
            self._register(i, in_dim, methods, config)
        self.one_hot_columns = self.tgt_columns[self._one_hot_indices]
        self.embedding_columns = self.tgt_columns[self._embed_indices]
        self._all_one_hot = len(self.one_hot_columns) == len(input_dims)
        self._all_embedding = len(self.embedding_columns) == len(input_dims)
        self.embedding_dropout = None
        if self.use_embedding and 0.0 < embedding_dropout < 1.0:
            self.embedding_dropout = Dropout(embedding_dropout)
        self._compile(loaders)

    @property
    def use_one_hot(self) -> bool:
        return len(self._one_hot_indices) > 0

    @property
    def use_embedding(self) -> bool:
        return len(self._embed_indices) > 0

    def forward(
        self,
        x_batch: torch.Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
    ) -> EncodingResult:
        keys = None
        if loader_name is not None:
            keys = self._get_cache_keys(loader_name)
        categorical_columns = x_batch[..., self.tgt_columns]
        if batch_indices is None or loader_name is None:
            self._oob_imputation(categorical_columns)
        # one hot
        if not self.use_one_hot:
            one_hot = None
        else:
            if keys is not None and batch_indices is not None:
                one_hot = getattr(self, keys["one_hot"])[batch_indices]
            else:
                one_hot_columns = categorical_columns
                if not self._all_one_hot:
                    one_hot_columns = one_hot_columns[..., self._one_hot_indices]
                one_hot_encodings = self._one_hot(one_hot_columns)
                one_hot = torch.cat(one_hot_encodings, dim=1)
        # embedding
        if not self._embed_indices:
            embedding = None
        else:
            if keys is None or batch_indices is None:
                indices = categorical_columns
            else:
                indices = getattr(self, keys["indices"])[batch_indices]
            if not self._all_embedding:
                indices = indices[..., self._embed_indices]
            embedding_encodings = self._embedding(indices)
            embedding = torch.cat(embedding_encodings, dim=1)
            if self.embedding_dropout is not None:
                embedding = self.embedding_dropout(embedding)
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
        self.one_hot_encoders.append(OneHot(in_dim))
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
        self.embeddings.append(Embedding(in_dim, out_dim, init_method, init_config))

    @staticmethod
    def _get_dim_sum(encodings: List[torch.Tensor], indices: List[int]) -> int:
        if not indices:
            return 0
        return sum([encodings[i].shape[1] for i in indices])

    def _oob_imputation(
        self,
        categorical_columns: torch.Tensor,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
    ) -> None:
        if loader_name is None or batch_indices is None:
            oob_mask = categorical_columns > self.bounds
        else:
            keys = self._get_cache_keys(loader_name)
            oob_mask = getattr(self, keys["oob"])[batch_indices]
        if torch.any(oob_mask):
            self.log_msg(  # type: ignore
                "out of bound occurred, "
                f"ratio : {torch.mean(oob_mask.to(torch.float32)).item():8.6f}",
                prefix=self.warning_prefix,
                verbose_level=5,
                msg_level=logging.WARNING,
            )
            # TODO : currently pytorch does not support onnx with bool masks
            #        in the future this line should be un-indented
            categorical_columns[oob_mask] = 0.0

    @staticmethod
    def _to_split(columns: torch.Tensor) -> List[torch.Tensor]:
        splits = columns.to(torch.long).t().split(1)
        return [split.view(-1) for split in splits]

    def _one_hot(self, one_hot_columns: torch.Tensor) -> List[torch.Tensor]:
        split = self._to_split(one_hot_columns)
        return [
            encoder(flat_feature)
            for encoder, flat_feature in zip(self.one_hot_encoders, split)
        ]

    def _embedding(self, indices_columns: torch.Tensor) -> List[torch.Tensor]:
        split = self._to_split(indices_columns)
        return [
            embedding(flat_feature)
            for embedding, flat_feature in zip(self.embeddings, split)
        ]

    @staticmethod
    def _get_cache_keys(name: str) -> Dict[str, str]:
        return {
            "one_hot": f"{name}_one_hot_cache",
            "indices": f"{name}_indices_cache",
            "oob": f"{name}_oob_cache",
        }

    def _compile(self, loaders: Dict[str, DataLoader]) -> None:
        for name, loader in loaders.items():
            categorical_features = []
            return_indices = loader.return_indices
            for a, b in loader:
                if return_indices:
                    x_batch, y_batch = a
                else:
                    x_batch, y_batch = a, b
                categorical_features.append(x_batch[..., self.tgt_columns])
            tensor = to_torch(np.vstack(categorical_features))
            keys = self._get_cache_keys(name)
            # compile oob
            indices = tensor.to(torch.long)
            self.register_buffer(keys["indices"], indices)
            self.register_buffer(keys["oob"], tensor > self.bounds)
            self._oob_imputation(
                tensor,
                batch_indices=np.arange(len(tensor)),
                loader_name=name,
            )
            # compile one hot
            if self.use_one_hot:
                one_hot_encodings = self._one_hot(tensor[..., self._one_hot_indices])
                one_hot_cache = torch.cat(one_hot_encodings, dim=1)
                self.register_buffer(keys["one_hot"], one_hot_cache)


__all__ = ["Encoder", "EncodingResult"]
