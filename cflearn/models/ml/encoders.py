import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import NamedTuple
from collections import defaultdict
from cfdata.tabular.misc import np_int_type

from ...data import MLLoader
from ...constants import INPUT_KEY
from ...misc.toolkit import to_torch
from ...misc.toolkit import Initializer
from ...modules.blocks import Lambda


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
        one_hot_fn = lambda column: F.one_hot(column, dim)
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
        embedding_fn = lambda column: F.embedding(column, self.weights)
        self.core = Lambda(embedding_fn, f"embedding: {in_dim} -> {out_dim}")
        self.in_dim, self.out_dim = in_dim, out_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.core(tensor)


class Encoder(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        input_dims: List[int],
        methods_list: List[Union[str, List[str]]],
        method_configs: List[Dict[str, Any]],
        categorical_columns: List[int],
        loaders: List[MLLoader],
    ):
        super().__init__()
        self._fe_init_method: Optional[str]
        self._fe_init_config: Optional[Dict[str, Any]]
        self._init_config(config)
        for loader in loaders:
            if loader.shuffle:
                raise ValueError("`loader` should not be shuffled in `Encoder`")
        self.merged_dim = 0
        self.one_hot_dim = 0
        self.embedding_dim = 0
        dims_tensor = torch.tensor(input_dims, dtype=torch.float32)
        self.register_buffer("input_dims", dims_tensor)
        self.tgt_columns = np.array(sorted(categorical_columns), np_int_type)
        self.merged_dims: Dict[int, int] = defaultdict(int)
        self.embeddings = nn.ModuleList()
        self.one_hot_encoders = nn.ModuleList()
        self._one_hot_indices: List[int] = []
        self._embed_indices: List[int] = []
        self._embed_dims: List[int] = []
        iterator = zip(input_dims, methods_list, method_configs)
        for i, (in_dim, methods, methods_config) in enumerate(iterator):
            if isinstance(methods, str):
                methods = [methods]
            self._register(i, in_dim, methods, methods_config)
        # fast embedding
        if self.use_embedding and self._use_fast_embed:
            if isinstance(self._unified_embed_dim, int):
                unified_dim = self._unified_embed_dim
            else:
                if self._unified_embed_dim == "max":
                    unified_dim = max(self._embed_dims)
                elif self._unified_embed_dim == "mean":
                    unified_dim = int(round(sum(self._embed_dims) / self.num_embedding))
                elif self._unified_embed_dim == "median":
                    half_idx = self.num_embedding // 2
                    sorted_dims = sorted(self._embed_dims)
                    if self.num_embedding % 2 != 0:
                        unified_dim = sorted_dims[half_idx]
                    else:
                        left = sorted_dims[half_idx - 1]
                        right = sorted_dims[half_idx]
                        unified_dim = int(round(0.5 * (left + right)))
                else:
                    raise NotImplementedError(
                        f"unified embedding dim '{self._unified_embed_dim}' "
                        "is not recognized"
                    )
            for i in self._embed_indices:
                self.merged_dims[i] += unified_dim
            embedding_dim_sum = unified_dim * self.num_embedding
            self.embedding_dim = embedding_dim_sum
            self.merged_dim += embedding_dim_sum
            if self._fe_init_method is None:
                self._fe_init_method = self._de_init_method
            if self._fe_init_config is None:
                self._fe_init_config = self._de_init_config
            assert isinstance(self._fe_init_config, dict)
            self.embeddings.append(
                Embedding(
                    sum(input_dims),
                    unified_dim,
                    self._fe_init_method,
                    self._fe_init_config,
                )
            )
            assert isinstance(self.input_dims, torch.Tensor)
            embed_dims_cumsum = self.input_dims[self._embed_indices].cumsum(0)[:-1]
            embed_dims_cumsum = embed_dims_cumsum.to(torch.long)
            self.register_buffer("embed_dims_cumsum", embed_dims_cumsum)
            if not self._recover_dim or len(set(self._embed_dims)) == 1:
                self.register_buffer("recover_indices", None)
            else:
                recover_indices: List[int] = []
                for i, dim in enumerate(self._embed_dims):
                    recover_indices.extend(i * unified_dim + j for j in range(dim))
                recover_indices_tensor = torch.tensor(recover_indices, dtype=torch.long)
                self.register_buffer("recover_indices", recover_indices_tensor)
        # embedding dropout
        self.embedding_dropout = None
        if self.use_embedding and 0.0 < self._embed_drop < 1.0:
            self.embedding_dropout = nn.Dropout(self._embed_drop)
        # compile
        self.one_hot_columns = self.tgt_columns[self._one_hot_indices]
        self.embedding_columns = self.tgt_columns[self._embed_indices]
        self._all_one_hot = len(self.one_hot_columns) == len(input_dims)
        self._all_embedding = len(self.embedding_columns) == len(input_dims)
        self._compile(loaders)

    @property
    def num_one_hot(self) -> int:
        return len(self._one_hot_indices)

    @property
    def num_embedding(self) -> int:
        return len(self._embed_indices)

    @property
    def use_one_hot(self) -> bool:
        return self.num_one_hot > 0

    @property
    def use_embedding(self) -> bool:
        return self.num_embedding > 0

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
            categorical_columns = self._oob_imputation(categorical_columns)
        use_cache = keys is not None and batch_indices is not None
        # one hot
        if not self.use_one_hot:
            one_hot = None
        else:
            if use_cache:
                one_hot = getattr(self, keys["one_hot"])[batch_indices]  # type: ignore
            else:
                one_hot_columns = categorical_columns
                if not self._all_one_hot:
                    one_hot_columns = one_hot_columns[..., self._one_hot_indices]
                one_hot = self._one_hot(one_hot_columns)
        # embedding
        if not self.use_embedding:
            embedding = None
        else:
            if not use_cache:
                indices = categorical_columns
            else:
                indices = getattr(self, keys["indices"])[batch_indices]  # type: ignore
            if not self._all_embedding:
                indices = indices[..., self._embed_indices]
            if not use_cache and self._use_fast_embed:
                indices[..., 1:] += self.embed_dims_cumsum
            embedding = self._embedding(indices.to(torch.long))
            if self.embedding_dropout is not None:
                embedding = self.embedding_dropout(embedding)
        return EncodingResult(one_hot, embedding)

    def _init_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._embed_drop = config.setdefault("embedding_dropout", 0.2)
        self._de_init_method = config.setdefault(
            "default_embedding_init_method",
            "truncated_normal",
        )
        self._de_init_config = config.setdefault(
            "default_embedding_init_config",
            {"mean": 0.0, "std": 0.02},
        )
        self._use_fast_embed = config.setdefault("use_fast_embedding", True)
        self._recover_dim = config.setdefault("recover_original_dim", False)
        # [ mean | median | max | int ]
        self._unified_embed_dim = config.setdefault("unified_embedding_dim", "max")
        self._fe_init_method = config.setdefault("fast_embedding_init_method", None)
        self._fe_init_config = config.setdefault("fast_embedding_init_config", None)

    def _register(
        self,
        i: int,
        in_dim: int,
        methods: List[str],
        methods_config: Dict[str, Any],
    ) -> None:
        for method in methods:
            attr = getattr(self, f"_register_{method}", None)
            if attr is None:
                msg = f"encoding method '{method}' is not implemented"
                raise NotImplementedError(msg)
            attr(i, in_dim, methods_config)

    def _register_one_hot(self, i: int, in_dim: int, _: Dict[str, Any]) -> None:
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
        if self._use_fast_embed:
            self._embed_dims.append(out_dim)
            if self._fe_init_method is None:
                self._fe_init_method = config.get("init_method")
            if self._fe_init_config is None:
                self._fe_init_config = config.get("init_config")
        else:
            init_method = config.setdefault("init_method", self._de_init_method)
            init_config = config.setdefault("init_config", self._de_init_config)
            self.merged_dims[i] += out_dim
            self.embedding_dim += out_dim
            self.merged_dim += out_dim
            self.embeddings.append(Embedding(in_dim, out_dim, init_method, init_config))

    def _oob_imputation(
        self,
        categorical_columns: torch.Tensor,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
    ) -> torch.Tensor:
        if loader_name is None or batch_indices is None:
            oob_mask = categorical_columns >= self.input_dims
        else:
            keys = self._get_cache_keys(loader_name)
            oob_mask = getattr(self, keys["oob"])[batch_indices].to(torch.bool)
        return torch.where(
            oob_mask,
            torch.zeros_like(categorical_columns),
            categorical_columns,
        )

    @staticmethod
    def _to_split(columns: torch.Tensor) -> List[torch.Tensor]:
        return list(columns.to(torch.long).t().unbind())

    def _one_hot(self, one_hot_columns: torch.Tensor) -> torch.Tensor:
        split = self._to_split(one_hot_columns)
        encodings = [
            encoder(flat_feature)
            for encoder, flat_feature in zip(self.one_hot_encoders, split)
        ]
        return torch.cat(encodings, dim=1)

    def _embedding(self, indices_columns: torch.Tensor) -> torch.Tensor:
        if self._use_fast_embed:
            embed_mat = self.embeddings[0](indices_columns)
            embed_mat = embed_mat.view(-1, self.embedding_dim)
            if not self._recover_dim or self.recover_indices is None:
                return embed_mat
            return embed_mat[..., self.recover_indices]
        split = self._to_split(indices_columns)
        encodings = [
            embedding(flat_feature)
            for embedding, flat_feature in zip(self.embeddings, split)
        ]
        return torch.cat(encodings, dim=1)

    @staticmethod
    def _get_cache_keys(name: str) -> Dict[str, str]:
        return {
            "one_hot": f"{name}_one_hot_cache",
            "indices": f"{name}_indices_cache",
            "oob": f"{name}_oob_cache",
        }

    def _compile(self, loaders: List[MLLoader]) -> None:
        for loader in loaders:
            name = loader.name
            if name is None:
                continue
            categorical_features = []
            for sample in loader:
                x_batch = sample[INPUT_KEY]
                categorical_features.append(x_batch[..., self.tgt_columns])
            tensor = to_torch(np.vstack(categorical_features))
            keys = self._get_cache_keys(name)
            # compile oob
            oob = (tensor >= self.input_dims).to(torch.float32)
            self.register_buffer(keys["oob"], oob)
            tensor = self._oob_imputation(
                tensor,
                batch_indices=np.arange(len(tensor)),
                loader_name=name,
            )
            # compile one hot
            if self.use_one_hot:
                one_hot_cache = self._one_hot(tensor[..., self._one_hot_indices])
                self.register_buffer(keys["one_hot"], one_hot_cache)
            # compile embedding
            if self.use_embedding and self._use_fast_embed:
                tensor[..., 1:] += self.embed_dims_cumsum
                indices = tensor.to(torch.long)
                self.register_buffer(keys["indices"], indices.to(torch.float32))


__all__ = ["Encoder", "EncodingResult"]
