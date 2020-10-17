import torch

import numpy as np
import torch.nn as nn

from typing import *
from cfdata.tabular import ColumnTypes
from cfdata.tabular import TabularData
from cftool.misc import LoggingMixin
from cftool.misc import register_core
from torch.optim import Optimizer
from abc import ABCMeta, abstractmethod

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..losses import *
from ..modules import *
from ..types import np_dict_type
from ..types import tensor_dict_type
from ..misc.toolkit import to_torch

model_dict: Dict[str, Type["ModelBase"]] = {}


def collate_np_dicts(ds: List[np_dict_type], axis: int = 0) -> np_dict_type:
    results = {}
    d0 = ds[0]
    for k in d0.keys():
        if not isinstance(d0[k], np.ndarray):
            continue
        arrays = []
        for rs in ds:
            array = rs[k]
            if len(array.shape) == 0:
                array = array.reshape([1])
            arrays.append(array)
        results[k] = np.concatenate(arrays, axis=axis)
    return results


def collate_tensor_dicts(ds: List[tensor_dict_type], dim: int = 0) -> tensor_dict_type:
    results = {}
    d0 = ds[0]
    for k in d0.keys():
        if not isinstance(d0[k], torch.Tensor):
            continue
        tensors = []
        for rs in ds:
            tensor = rs[k]
            if len(tensor.shape) == 0:
                tensor = tensor.view([1])
            tensors.append(tensor)
        results[k] = torch.cat(tensors, dim=dim)
    return results


class SplitFeatures(NamedTuple):
    categorical: Union[torch.Tensor, tensor_dict_type, None]
    numerical: Union[torch.Tensor, None]

    @property
    def stacked_categorical(self) -> Optional[torch.Tensor]:
        if not isinstance(self.categorical, dict):
            return self.categorical
        return torch.cat([self.categorical[k] for k in sorted(self.categorical)], dim=1)

    def merge(self) -> torch.Tensor:
        categorical = self.stacked_categorical
        if categorical is None:
            assert self.numerical is not None
            return self.numerical
        if self.numerical is None:
            assert categorical is not None
            return categorical
        return torch.cat([self.numerical, categorical], dim=1)


class ModelBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self._pipeline_config = pipeline_config
        self.config = pipeline_config.setdefault("model_config", {})
        self._preset_config(tr_data)
        self._init_config(tr_data)
        self._init_loss(tr_data)
        # encoders
        excluded = 0
        ts_indices = tr_data.ts_indices
        recognizers = tr_data.recognizers
        encoders: Dict[str, EncoderStack] = {}
        self.numerical_columns_mapping = {}
        self.categorical_columns_mapping = {}
        for idx, recognizer in recognizers.items():
            if idx == -1:
                continue
            if not recognizer.info.is_valid or idx in ts_indices:
                excluded += 1
            elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                self.numerical_columns_mapping[idx] = idx - excluded
            else:
                self._init_encoder(idx, encoders)
                self.categorical_columns_mapping[idx] = idx - excluded
        self.encoders = nn.ModuleDict(encoders)
        self._categorical_dim = sum(encoder.dim for encoder in self.encoders.values())
        self._numerical_columns = sorted(self.numerical_columns_mapping.values())

    @property
    @abstractmethod
    def input_sample(self) -> tensor_dict_type:
        x = self.tr_data.processed.x[:2]
        y = self.tr_data.processed.y[:2]
        x, y = map(to_torch, [x, y])
        return {"x_batch": x, "y_batch": y}

    @abstractmethod
    def forward(self, batch: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        # batch will have `categorical`, `numerical` and `labels` keys
        # requires returning `predictions` key
        pass

    def loss_function(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
    ) -> Dict[str, torch.Tensor]:
        # requires returning `loss` key
        y_batch = batch["y_batch"]
        if self.tr_data.is_clf:
            y_batch = y_batch.view(-1)
        predictions = forward_results["predictions"]
        sample_weights = forward_results.get("batch_sample_weights")
        losses = self.loss(predictions, y_batch)
        if sample_weights is None:
            return {"loss": losses.mean()}
        return {"loss": (losses * sample_weights.to(losses.device)).mean()}

    @property
    def num_history(self) -> int:
        num_history = 1
        if self.tr_data.is_ts:
            sampler_config = self._pipeline_config["sampler_config"]
            aggregation_config = sampler_config.get("aggregation_config", {})
            num_history = aggregation_config.get("num_history")
            if num_history is None:
                raise ValueError(
                    "please provide `num_history` in `aggregation_config` "
                    "in `cflearn.make` for time series tasks."
                )
        return num_history

    @property
    def merged_dim(self) -> int:
        merged_dim = self._categorical_dim + len(self._numerical_columns)
        return merged_dim * self.num_history

    @property
    def encoding_dims(self) -> Dict[str, int]:
        encoding_dims: Dict[str, int] = {}
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

    def _preset_config(self, tr_data: TabularData) -> None:
        pass

    def _init_config(self, tr_data: TabularData) -> None:
        self.tr_data = tr_data
        self._loss_config = self.config.setdefault("loss_config", {})
        # TODO : optimize encodings by pre-calculate one-hot encodings in Trainer
        self._encoding_methods = self.config.setdefault("encoding_methods", {})
        self._encoding_configs = self.config.setdefault("encoding_configs", {})
        self._default_encoding_configs = self.config.setdefault(
            "default_encoding_configs", {}
        )
        self._default_encoding_method = self.config.setdefault(
            "default_encoding_method", "embedding"
        )

    def _init_input_config(self) -> None:
        self._fc_in_dim: int = self.config.get("fc_in_dim")
        self._fc_out_dim: int = self.config.get("fc_out_dim")
        self.out_dim = max(self.tr_data.num_classes, 1)
        if self._fc_in_dim is None:
            self._fc_in_dim = self.merged_dim
        if self._fc_out_dim is None:
            self._fc_out_dim = self.out_dim

    def _init_loss(self, tr_data: TabularData) -> None:
        if tr_data.is_reg:
            self.loss: nn.Module = nn.L1Loss(reduction="none")
        else:
            self.loss = FocalLoss(self._loss_config, reduction="none")

    def _init_encoder(self, idx: int, encoders: Dict[str, EncoderStack]) -> None:
        methods = self._encoding_methods.setdefault(idx, self._default_encoding_method)
        config = self._encoding_configs.setdefault(idx, self._default_encoding_configs)
        num_values = self.tr_data.recognizers[idx].num_unique_values
        if isinstance(methods, str):
            methods = [methods]
        local_encoders = [
            encoder_dict[method](idx, num_values, config) for method in methods
        ]
        encoders[str(idx)] = EncoderStack(*local_encoders)

    def _optimizer_step(
        self,
        optimizers: Dict[str, Optimizer],
        grad_scalar: Optional["amp.GradScaler"],  # type: ignore
    ) -> None:
        for opt in optimizers.values():
            if grad_scalar is None:
                opt.step()
            else:
                grad_scalar.step(opt)
                grad_scalar.update()
            opt.zero_grad()

    @staticmethod
    def _switch_requires_grad(
        params: List[torch.nn.Parameter],
        requires_grad: bool,
    ) -> None:
        for param in params:
            param.requires_grad_(requires_grad)

    def _split_features(
        self,
        x_batch: torch.Tensor,
        *,
        return_all_encodings: bool = False,
    ) -> SplitFeatures:
        categorical_columns = []
        for idx_str in sorted(self.encoders):
            encoder = self.encoders[idx_str]
            mapping_idx = self.categorical_columns_mapping[int(idx_str)]
            categorical_columns.append(
                encoder(x_batch[..., mapping_idx], return_all=return_all_encodings)
            )

        categorical: Union[torch.Tensor, tensor_dict_type, None]
        if not categorical_columns:
            categorical = None
        elif not return_all_encodings:
            categorical = torch.cat(categorical_columns, dim=1)
        else:
            categorical = collate_tensor_dicts(categorical_columns, dim=1)

        numerical = (
            None
            if not self._numerical_columns
            else x_batch[..., self._numerical_columns]
        )

        return SplitFeatures(categorical, numerical)

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device))

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global model_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, model_dict, before_register=before)


__all__ = [
    "collate_np_dicts",
    "collate_tensor_dicts",
    "SplitFeatures",
    "ModelBase",
]
