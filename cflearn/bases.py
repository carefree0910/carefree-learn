import os
import json
import torch
import pprint
import logging

import numpy as np
import torch.nn as nn

from typing import *
from cftool.ml import *
from cftool.misc import *
from cfdata.tabular import *
from tqdm import tqdm

from abc import ABCMeta, abstractmethod

from .modules import *
from .misc.toolkit import *

data_type = Union[np.ndarray, List[List[float]], str]
model_dict: Dict[str, Type["ModelBase"]] = {}


class SplitFeatures(NamedTuple):
    categorical: Union[torch.Tensor, Dict[str, torch.Tensor], None]
    numerical: Union[torch.Tensor, None]

    @property
    def stacked_categorical(self) -> torch.Tensor:
        if not isinstance(self.categorical, dict):
            return self.categorical
        return torch.cat([self.categorical[k] for k in sorted(self.categorical)], dim=1)

    def merge(self) -> torch.Tensor:
        categorical = self.stacked_categorical
        if categorical is None:
            return self.numerical
        if self.numerical is None:
            return categorical
        return torch.cat([self.numerical, categorical], dim=1)


class ModelBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(self,
                 config: Dict[str, Any],
                 tr_data: TabularData,
                 device: torch.device):
        super().__init__()
        self._preset_config(config, tr_data)
        self._init_config(config, tr_data)
        self.device = device
        # encoders
        excluded = 0
        recognizers = tr_data.recognizers
        self.encoders = {}
        self.numerical_columns_mapping = {}
        self.categorical_columns_mapping = {}
        for idx, recognizer in recognizers.items():
            if idx == -1:
                continue
            if not recognizer.info.is_valid:
                excluded += 1
            elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                self.numerical_columns_mapping[idx] = idx - excluded
            else:
                self._init_encoder(idx)
                self.categorical_columns_mapping[idx] = idx - excluded
        if self.encoders:
            self.encoders = nn.ModuleDict(self.encoders)
        self._categorical_dim = sum(encoder.dim for encoder in self.encoders.values())
        self._numerical_columns = sorted(self.numerical_columns_mapping.values())

    @abstractmethod
    def forward(self,
                batch: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        # batch will have `categorical`, `numerical` and `labels` keys
        # requires returning `predictions` key
        pass

    def loss_function(self,
                      batch: Dict[str, torch.Tensor],
                      forward_results: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, float]]:
        # requires returning `loss` key
        y_batch = batch["y_batch"]
        if self.tr_data.is_clf:
            y_batch = y_batch.view(-1)
        predictions = forward_results["predictions"]
        return {"loss": self.loss(predictions, y_batch)}

    @property
    def merged_dim(self):
        return self._categorical_dim + len(self._numerical_columns)

    @property
    def encoding_dims(self) -> Dict[str, int]:
        encoding_dims = {}
        for encoder_stack in self.encoders.values():
            dims = encoder_stack.dims
            if not encoding_dims:
                encoding_dims = dims
            else:
                for k, v in dims.items():
                    encoding_dims[k] += v
        return encoding_dims

    @property
    def categorical_dims(self) -> Dict[int, int]:
        dims = {}
        for idx_str in sorted(self.encoders):
            idx = int(idx_str)
            true_idx = self.categorical_columns_mapping[idx]
            dims[true_idx] = self.encoders[idx_str].dim
        return dims

    def _preset_config(self,
                       config: Dict[str, Any],
                       tr_data: TabularData):
        pass

    def _init_config(self,
                     config: Dict[str, Any],
                     tr_data: TabularData):
        self.tr_data = tr_data
        self.config = config
        # TODO : optimize encodings by pre-calculate one-hot encodings in Wrapper
        self._encoding_methods = self.config.setdefault("encoding_methods", {})
        self._encoding_configs = self.config.setdefault("encoding_configs", {})
        self._default_encoding_method = self.config.setdefault("default_encoding_method", "embedding")
        self._init_loss(config, tr_data)

    def _init_loss(self,
                   config: Dict[str, Any],
                   tr_data: TabularData):
        if tr_data.is_reg:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def _init_encoder(self, idx: int):
        methods = self._encoding_methods.setdefault(idx, self._default_encoding_method)
        config = self._encoding_configs.setdefault(idx, {})
        num_values = self.tr_data.recognizers[idx].num_unique_values
        if isinstance(methods, str):
            methods = [methods]
        encoders = [encoder_dict[method](idx, num_values, config) for method in methods]
        self.encoders[str(idx)] = EncoderStack(*encoders)

    @staticmethod
    def _collate_tensor_dicts(ds: List[Dict[str, torch.Tensor]],
                              dim: int = 0) -> Dict[str, torch.Tensor]:
        return {k: torch.cat([rs[k] for rs in ds], dim=dim) for k in ds[0].keys()}

    @staticmethod
    def to_prob(raw: np.ndarray) -> np.ndarray:
        return nn.functional.softmax(torch.from_numpy(raw), dim=1).numpy()

    def _split_features(self,
                        x_batch: torch.Tensor,
                        *,
                        return_all_encodings: bool = False) -> SplitFeatures:
        categorical_columns = []
        for idx_str in sorted(self.encoders):
            encoder = self.encoders[idx_str]
            mapping_idx = self.categorical_columns_mapping[int(idx_str)]
            categorical_columns.append(encoder(x_batch[..., mapping_idx], return_all=return_all_encodings))
        if not categorical_columns:
            categorical = None
        elif not return_all_encodings:
            categorical = torch.cat(categorical_columns, dim=1)
        else:
            categorical = self._collate_tensor_dicts(categorical_columns, dim=1)
        numerical = None if not self._numerical_columns else x_batch[..., self._numerical_columns]
        return SplitFeatures(categorical, numerical)

    def get_split(self,
                  processed: np.ndarray,
                  device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device))

    @classmethod
    def register(cls, name: str):
        global model_dict
        def before(cls_): cls_.__identifier__ = name
        return register_core(name, model_dict, before_register=before)
