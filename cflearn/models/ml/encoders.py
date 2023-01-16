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
from cftool.array import arr_type
from cftool.array import to_torch

from ..protocols.ml import EncodingResult
from ...data import MLLoader
from ...constants import INPUT_KEY
from ...misc.toolkit import Initializer
from ...modules.blocks import Lambda


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
        self.in_dim, self.out_dim = in_dim, out_dim

    def extra_repr(self) -> str:
        return f"{self.in_dim} -> {self.out_dim}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.embedding(tensor, self.weights)


class EncodingSettings(NamedTuple):
    """
    Encoder settings.

    Properties
    ----------
    dim (int) : number of different values of this categorical column.
    methods (str | List[str]) : encoding methods to use for each categorical column.
        * if List[str] is provided and its length > 1, then multiple encoding methods will be used.
    method_configs (Dict[str, Any]) : (flattened) configs of the corresponding encoding methods.
        * even if multiple methods are used, `method_configs` should still be 'flattened'

    """

    dim: int
    methods: Union[str, List[str]] = "embedding"
    method_configs: Optional[Dict[str, Any]] = None

    @property
    def use_one_hot(self) -> bool:
        if self.methods == "one_hot":
            return True
        if isinstance(self.methods, list) and "one_hot" in self.methods:
            return True
        return False

    @property
    def use_embedding(self) -> bool:
        if self.methods == "embedding":
            return True
        if isinstance(self.methods, list) and "embedding" in self.methods:
            return True
        return False


class Encoder(nn.Module):
    def __init__(
        self,
        settings: Dict[int, EncodingSettings],
        *,
        config: Optional[Dict[str, Any]] = None,
        loaders: Optional[List[MLLoader]] = None,
    ):
        """
        Encoder to perform differentiable categorical encodings.

        Terminology
        ----------
        `i`   : indicates the {i}th element of `columns`.
        `idx` : indicates the actual column index.

        -> So basically, we have self.columns[i] = idx.

        Parameters
        ----------
        settings (Dict[`idx`, EncoderSettings]) : encoding settings of each categorical column.
            * key: `idx`.
            * value: `EncoderSettings`.
        config (Dict[str, Any]) : configs of the encoder.
            * embedding_dropout (float) : dropout rate for embedding.
                * default : 0.2
            * default_embedding_init_method (str) : default init method for embedding.
                * default : "truncated_normal"
            * default_embedding_init_config (Dict[str, Any]) : default init config for embedding.
                * default : {"mean": 0, "std": 0.02}
            * use_fast_embedding (bool) : whether to use fast embedding.
                * default : True
            * recover_original_dim (bool) : whether to recover original dim when fast embedding is applied.
                * default : False
            * unified_embedding_dim (str | int) : unified embedding dim used in fast embedding.
                * default : "max"
            * fast_embedding_init_method (str | None) : init method for fast embedding.
                * default : None
            * fast_embedding_init_config (Dict[str, Any] | None) : init config for fast embedding.
                * default : None
        loaders (List[MLLoader]) : internally used by `carefree-learn`.

        Properties
        ----------
        columns (List[`idx`]) : sorted categorical columns.
        input_dims (tensor) : number of different values of each categorical column.
            * len(input_dims) = len(columns)
            * input_dims[i] indicates the number of different values of the {columns[i]}th column.
        _one_hot_indices (List[`i`]) : indices that are encoded with one-hot.
        _one_hot_columns (List[`idx`]) : columns that are encoded with one-hot.
        _embed_indices (List[`i`]) : indices that are encoded with embedding.
        _embed_columns (List[`idx`]) : indices that are encoded with embedding.
        _embed_dims (Dict[`idx`, int]) : dims of embedding used for each categorical column.
            * key : `idx`.
            * value : embed dimension of the embedding module.
            * len(_embed_dims) should be = len(_embed_indices).
        merged_dims (Dict[`idx`, int])
            * key : `idx`.
            * value : final encoded dim of the {key}th column.
        embeddings (nn.ModuleDict) : collection of embedding modules.
            * key : str(`idx`).
            * value : embedding module of the {key}th column.
        one_hot_encoders (nn.ModuleDict) : collection of one-hot encoder modules.
            * key : str(`idx`).
            * value : one-hot encoder module of the {key}th column.

        """
        super().__init__()
        loaders = loaders or []
        self._fe_init_method: Optional[str]
        self._fe_init_config: Optional[Dict[str, Any]]
        self._init_config(config or {})
        for loader in loaders:
            if loader.shuffle:
                raise ValueError("`loader` should not be shuffled in `Encoder`")
        self.merged_dim = 0
        self.one_hot_dim = 0
        self.embedding_dim = 0
        self.columns = sorted(settings)
        self.tgt_columns = np.array(self.columns, np.int64)
        self.register_buffer(
            "input_dims",
            torch.tensor(
                [settings[idx].dim for idx in self.columns],
                dtype=torch.float32,
            ),
        )
        self.merged_dims: Dict[int, int] = defaultdict(int)
        self.embeddings = nn.ModuleDict()
        self.one_hot_encoders = nn.ModuleDict()
        self._one_hot_indices: List[int] = []
        self._one_hot_columns: List[int] = []
        self._embed_indices: List[int] = []
        self._embed_columns: List[int] = []
        self._embed_dims: Dict[int, int] = {}
        for i, idx in enumerate(self.columns):
            self._register(i, idx, settings[idx])
        # fast embedding
        self.unified_dim = 0
        if self.use_embedding and self._use_fast_embed:
            sorted_dims = sorted(self._embed_dims.values())
            max_dim = sorted_dims[-1]
            if isinstance(self._unified_embed_dim, int):
                unified_dim = self._unified_embed_dim
            else:
                if self._unified_embed_dim == "max":
                    unified_dim = max_dim
                elif self._unified_embed_dim == "mean":
                    unified_dim = int(round(sum(sorted_dims) / self.num_embedding))
                elif self._unified_embed_dim == "median":
                    half_idx = self.num_embedding // 2
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
            if self._recover_dim and unified_dim < max_dim:
                raise ValueError(
                    f"unified embedding dim {unified_dim} is smaller than "
                    f"max embedding dim {max_dim}, which is not allowed "
                    f"when `_recover_dim` is set to True"
                )
            self.unified_dim = unified_dim
            embedding_dim_sum = 0
            for idx in self._embed_columns:
                original_dim = self._embed_dims[idx]
                final_dim = original_dim if self._recover_dim else unified_dim
                self.merged_dims[idx] += final_dim
                embedding_dim_sum += final_dim
            self.embedding_dim = embedding_dim_sum
            self.merged_dim += embedding_dim_sum
            if self._fe_init_method is None:
                self._fe_init_method = self._de_init_method
            if self._fe_init_config is None:
                self._fe_init_config = self._de_init_config
            assert isinstance(self._fe_init_config, dict)
            embed_input_dims = [settings[idx].dim for idx in self._embed_columns]
            self.embeddings["-1"] = Embedding(
                sum(embed_input_dims),
                unified_dim,
                self._fe_init_method,
                self._fe_init_config,
            )
            embed_input_dims_cumsum = torch.tensor(embed_input_dims).cumsum(0)[:-1]
            embed_input_dims_cumsum = embed_input_dims_cumsum.to(torch.long)
            self.register_buffer("embed_input_dims_cumsum", embed_input_dims_cumsum)
            if not self._recover_dim or len(set(self._embed_dims.values())) == 1:
                self.register_buffer("recover_indices", None)
            else:
                recover_indices: List[int] = []
                for i, idx in enumerate(self._embed_columns):
                    dim = self._embed_dims[idx]
                    recover_indices.extend(i * unified_dim + j for j in range(dim))
                recover_indices_tensor = torch.tensor(recover_indices, dtype=torch.long)
                self.register_buffer("recover_indices", recover_indices_tensor)
        # embedding dropout
        self.embedding_dropout = None
        if self.use_embedding and 0.0 < self._embed_drop < 1.0:
            self.embedding_dropout = nn.Dropout(self._embed_drop)
        # compile
        self._all_one_hot = len(self._one_hot_indices) == len(settings)
        self._all_embedding = len(self._embed_indices) == len(settings)
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
            use_cache = use_cache and self._use_fast_embed
            if not use_cache:
                indices = categorical_columns
            else:
                indices = getattr(self, keys["indices"])[batch_indices]  # type: ignore
            if not self._all_embedding:
                indices = indices[..., self._embed_indices]
            if not use_cache and self._use_fast_embed:
                indices[..., 1:] += self.embed_input_dims_cumsum
            embedding = self._embedding(indices.to(torch.long))
            if self.embedding_dropout is not None:
                embedding = self.embedding_dropout(embedding)
        return EncodingResult(one_hot, embedding)

    def encode(self, column: arr_type, column_idx: int) -> EncodingResult:
        if column_idx not in self.columns:
            return EncodingResult(None, None)
        if isinstance(column, np.ndarray):
            column = to_torch(column)
        column = column.to(torch.long).view(-1)
        # one hot
        if not self.use_one_hot or column_idx not in self._one_hot_columns:
            one_hot = None
        else:
            one_hot = self.one_hot_encoders[str(column_idx)](column)
        # embedding
        if not self.use_embedding or column_idx not in self._embed_columns:
            embedding = None
        else:
            indices = column.clone()
            if not self._use_fast_embed:
                embedding = self.embeddings[str(column_idx)](indices)
            else:
                i_embedding = self._embed_columns.index(column_idx)
                if i_embedding > 0:
                    indices += self.embed_input_dims_cumsum[i_embedding - 1]
                embedding = self.embeddings["-1"](indices)
                if self._recover_dim:
                    dim = self._embed_dims[column_idx]
                    embedding = embedding[..., list(range(dim))]
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

    def _register(self, i: int, idx: int, setting: EncodingSettings) -> None:
        methods = setting.methods
        if isinstance(methods, str):
            methods = [methods]
        for method in methods:
            attr = getattr(self, f"_register_{method}", None)
            if attr is None:
                msg = f"encoding method '{method}' is not implemented"
                raise NotImplementedError(msg)
            attr(i, idx, setting.dim, setting.method_configs or {})

    def _register_one_hot(
        self,
        i: int,
        idx: int,
        in_dim: int,
        _: Dict[str, Any],
    ) -> None:
        self.one_hot_encoders[str(idx)] = OneHot(in_dim)
        self._one_hot_indices.append(i)
        self._one_hot_columns.append(idx)
        self.merged_dims[idx] += in_dim
        self.one_hot_dim += in_dim
        self.merged_dim += in_dim

    @staticmethod
    def _get_embed_key(num: int) -> str:
        return f"embedding_weight_{num}"

    def _register_embedding(
        self,
        i: int,
        idx: int,
        in_dim: int,
        config: Dict[str, Any],
    ) -> None:
        self._embed_indices.append(i)
        self._embed_columns.append(idx)
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
            self._embed_dims[idx] = out_dim
            if self._fe_init_method is None:
                self._fe_init_method = config.get("init_method")
            if self._fe_init_config is None:
                self._fe_init_config = config.get("init_config")
        else:
            init_method = config.setdefault("init_method", self._de_init_method)
            init_config = config.setdefault("init_config", self._de_init_config)
            embedding = Embedding(in_dim, out_dim, init_method, init_config)
            self.embeddings[str(idx)] = embedding
            self.merged_dims[idx] += out_dim
            self.embedding_dim += out_dim
            self.merged_dim += out_dim

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
        return list(columns.to(torch.long).unbind(dim=-1))

    def _one_hot(self, one_hot_columns: torch.Tensor) -> torch.Tensor:
        split = self._to_split(one_hot_columns)
        encodings = [
            self.one_hot_encoders[str(self._one_hot_columns[i])](flat_feature)
            for i, flat_feature in enumerate(split)
        ]
        return torch.cat(encodings, dim=-1)

    def _embedding(self, indices_columns: torch.Tensor) -> torch.Tensor:
        if self._use_fast_embed:
            embed_mat = self.embeddings["-1"](indices_columns)
            embed_mat = embed_mat.view(
                *indices_columns.shape[:-1],
                self.num_embedding * self.unified_dim,
            )
            if not self._recover_dim or self.recover_indices is None:
                return embed_mat
            return embed_mat[..., self.recover_indices]
        split = self._to_split(indices_columns)
        encodings = [
            self.embeddings[str(self._embed_columns[i])](flat_feature)
            for i, flat_feature in enumerate(split)
        ]
        return torch.cat(encodings, dim=-1)

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
                tensor[..., 1:] += self.embed_input_dims_cumsum
                indices = tensor.to(torch.long)
                self.register_buffer(keys["indices"], indices.to(torch.float32))


__all__ = [
    "Encoder",
    "EncodingSettings",
]
