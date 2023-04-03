import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import NamedTuple
from cftool.array import to_torch
from cftool.types import arr_type

from ...schema import MLEncoderSettings
from ...schema import MLGlobalEncoderSettings
from ...misc.toolkit import shallow_copy_dict
from ...misc.toolkit import Initializer
from ...modules.blocks import Lambda


class OneHot(Lambda):
    def __init__(self, dim: int):
        self.dim = dim
        one_hot_fn = lambda column: F.one_hot(column, dim)
        super().__init__(one_hot_fn, f"one_hot_{dim}")

    def forward(self, net: Tensor) -> Tensor:
        return super().forward(net).to(torch.float32)


class EmbeddingConfig(NamedTuple):
    out_dim: int
    init_method: Optional[str]
    init_config: Dict[str, Any]
    dropout: float


class Embedding(nn.Module):
    def __init__(self, in_dim: int, config: EmbeddingConfig):
        super().__init__()
        weights = torch.empty(in_dim, config.out_dim)
        if config.init_method is None:
            nn.init.normal_(weights)
        else:
            Initializer(config.init_config).initialize(weights, config.init_method)
        self.weights = nn.Parameter(weights)
        if config.dropout <= 0.0 or 1.0 <= config.dropout:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(config.dropout, inplace=True)
        self.in_dim = in_dim
        self.out_dim = config.out_dim

    def extra_repr(self) -> str:
        return f"{self.in_dim} -> {self.out_dim}"

    def forward(self, net: Tensor) -> Tensor:
        net = F.embedding(net, self.weights)
        if self.dropout is not None:
            net = self.dropout(net)
        return net


class EncodingResult(NamedTuple):
    indices: Tensor
    one_hot: Optional[Tensor]
    embedding: Optional[Tensor]

    @property
    def merged(self) -> Optional[Tensor]:
        if self.one_hot is None and self.embedding is None:
            return None
        if self.one_hot is None:
            assert self.embedding is not None
            return self.embedding
        if self.embedding is None:
            assert self.one_hot is not None
            return self.one_hot
        return torch.cat([self.one_hot, self.embedding], dim=-1)


def get_embedding_config(
    in_dim: int,
    kw: Dict[str, Any],
    global_encoder_settings: Optional[MLGlobalEncoderSettings],
) -> EmbeddingConfig:
    ges = global_encoder_settings or MLGlobalEncoderSettings()
    default_out_dim: Union[str, int]
    if ges.embedding_dim is None:
        default_out_dim = "auto"
    else:
        default_out_dim = ges.embedding_dim
    # get out_dim
    out_dim = kw.get("out_dim", default_out_dim)
    if isinstance(out_dim, int):
        out_dim = out_dim
    elif out_dim == "log":
        out_dim = math.ceil(math.log2(in_dim))
    elif out_dim == "sqrt":
        out_dim = math.ceil(math.sqrt(in_dim))
    elif out_dim == "auto":
        out_dim = max(4, min(8, math.ceil(math.log2(in_dim))))
    else:
        raise ValueError(f"embedding dim '{out_dim}' is not defined")
    # get init arguments
    init_method = kw.get("init_method", "truncated_normal")
    init_config = kw.get("init_config", {"mean": 0.0, "std": 0.02})
    # get dropout
    dropout = kw.get(
        "dropout",
        0.1 if ges.embedding_dropout is None else ges.embedding_dropout,
    )
    # return
    return EmbeddingConfig(out_dim, init_method, init_config, dropout)


def to_split(columns: Tensor) -> List[Tensor]:
    return list(columns.unbind(dim=-1))


class Encoder(nn.Module):
    dims: Tensor

    def __init__(
        self,
        settings: Dict[str, MLEncoderSettings],
        global_encoder_settings: Optional[MLGlobalEncoderSettings] = None,
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.one_hot_encoders = nn.ModuleDict()
        dims = []
        self.tgt_columns = []
        self.one_hot_columns = []
        self.embedding_columns = []
        self.one_hot_dim = 0
        self.embedding_dim = 0
        self.dim_increment = 0
        for str_idx in sorted(settings):
            idx = int(str_idx)
            setting = settings[str_idx]
            dim = setting.dim
            dims.append(dim)
            self.tgt_columns.append(idx)
            if setting.use_one_hot:
                self.one_hot_columns.append(idx)
                self.one_hot_encoders[str_idx] = OneHot(dim)
                self.one_hot_dim += dim
                self.dim_increment += dim - 1
            if setting.use_embedding:
                self.embedding_columns.append(idx)
                kw = shallow_copy_dict(setting.method_configs or {})
                config = get_embedding_config(dim, kw, global_encoder_settings)
                self.embeddings[str_idx] = Embedding(dim, config)
                self.embedding_dim += config.out_dim
                self.dim_increment += config.out_dim - 1
        # setup useful properties
        self.categorical_dim = self.one_hot_dim + self.embedding_dim
        self.num_one_hot = len(self.one_hot_encoders)
        self.num_embedding = len(self.embeddings)
        self.use_one_hot = self.num_one_hot > 0
        self.use_embedding = self.num_embedding > 0
        self.is_empty = not self.use_one_hot and not self.use_embedding
        self.all_one_hot = len(self.one_hot_columns) == len(self.tgt_columns)
        self.all_embedding = len(self.embedding_columns) == len(self.tgt_columns)
        self.register_buffer("dims", torch.tensor(dims, dtype=torch.float32))

    # api

    # encode all columns
    def forward(self, x_batch: Tensor) -> EncodingResult:
        # extract categorical
        categorical_columns = x_batch[..., self.tgt_columns]
        # oob imputation
        oob_mask = categorical_columns >= self.dims
        if oob_mask.any().item():
            categorical_columns = torch.where(
                oob_mask,
                torch.zeros_like(categorical_columns),
                categorical_columns,
            )
        # turn to long indices
        indices = categorical_columns.to(torch.long)
        # one hot
        if not self.use_one_hot:
            one_hot = None
        else:
            if self.all_one_hot:
                one_hot_indices = indices
            else:
                one_hot_indices = indices[..., self.one_hot_columns]
            one_hot = self._one_hot(one_hot_indices)
        # embedding
        if not self.use_embedding:
            embedding = None
        else:
            if self.all_embedding:
                embedding_indices = indices
            else:
                embedding_indices = indices[..., self.embedding_columns]
            embedding = self._embedding(embedding_indices)
        # return
        return EncodingResult(indices, one_hot, embedding)

    # encode single column
    def encode(self, column: arr_type, column_idx: int) -> EncodingResult:
        if column_idx not in self.tgt_columns:
            return EncodingResult(None, None, None)
        if isinstance(column, np.ndarray):
            column = to_torch(column)
        indices = column.to(torch.long)
        flat_indices = indices.view(-1)
        # one hot
        if not self.use_one_hot or column_idx not in self.one_hot_columns:
            one_hot = None
        else:
            one_hot = self.one_hot_encoders[str(column_idx)](flat_indices)
        # embedding
        if not self.use_embedding or column_idx not in self.embedding_columns:
            embedding = None
        else:
            embedding_indices = flat_indices.clone()
            embedding = self.embeddings[str(column_idx)](embedding_indices)
            if self.embedding_dropout is not None:
                embedding = self.embedding_dropout(embedding)
        return EncodingResult(indices, one_hot, embedding)

    # internal

    def _one_hot(self, one_hot_indices: Tensor) -> Tensor:
        split = to_split(one_hot_indices)
        encodings = [
            self.one_hot_encoders[str(self.one_hot_columns[i])](flat_feature)
            for i, flat_feature in enumerate(split)
        ]
        return torch.cat(encodings, dim=-1)

    def _embedding(self, embedding_indices: Tensor) -> Tensor:
        split = to_split(embedding_indices)
        encodings = [
            self.embeddings[str(self.embedding_columns[i])](flat_feature)
            for i, flat_feature in enumerate(split)
        ]
        return torch.cat(encodings, dim=-1)


__all__ = [
    "Encoder",
    "EncodingResult",
]
