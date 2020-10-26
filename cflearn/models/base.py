import torch
import pprint

import numpy as np
import torch.nn as nn

from typing import *
from cfdata.tabular import ColumnTypes
from cfdata.tabular import DataLoader
from cfdata.tabular import TabularData
from cftool.misc import register_core
from cftool.misc import timing_context
from cftool.misc import LoggingMixin
from torch.optim import Optimizer
from abc import ABCMeta, abstractmethod

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..losses import *
from ..modules import *
from ..types import tensor_dict_type
from ..misc.toolkit import to_torch

model_dict: Dict[str, Type["ModelBase"]] = {}


class SplitFeatures(NamedTuple):
    categorical: Optional[EncodingResult]
    numerical: Optional[torch.Tensor]

    def merge(self) -> torch.Tensor:
        categorical = self.categorical.merged
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
        tr_loader: DataLoader,
        cv_data: TabularData,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        device: torch.device,
        *,
        use_tqdm: bool,
    ):
        super().__init__()
        self.ema: Optional[EMA] = None
        self.device = device
        self.use_tqdm = use_tqdm
        self._pipeline_config = pipeline_config
        self.config = pipeline_config.setdefault("model_config", {})
        self.tr_loader = tr_loader
        self.tr_data = tr_loader.data
        self.cv_data = cv_data
        self.tr_weights, self.cv_weights = tr_weights, cv_weights
        self._preset_config()
        self._init_config()
        self._init_loss()
        # encoder
        excluded = 0
        self.numerical_columns_mapping = {}
        self.categorical_columns_mapping = {}
        categorical_dims = []
        encoding_methods = []
        encoding_configs = []
        true_categorical_columns = []
        if self.tr_data.is_simplify:
            for idx in range(self.tr_data.raw_dim):
                self.numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.tr_data.ts_indices
            recognizers = self.tr_data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    self.numerical_columns_mapping[idx] = idx - excluded
                else:
                    str_idx = str(idx)
                    categorical_dims.append(
                        self.tr_data.recognizers[idx].num_unique_values
                    )
                    encoding_methods.append(
                        self._encoding_methods.setdefault(
                            str_idx, self._default_encoding_method
                        )
                    )
                    encoding_configs.append(
                        self._encoding_configs.setdefault(
                            str_idx, self._default_encoding_configs
                        )
                    )
                    true_idx = idx - excluded
                    true_categorical_columns.append(true_idx)
                    self.categorical_columns_mapping[idx] = true_idx
        if not true_categorical_columns:
            self.encoder = None
        else:
            self.encoder = Encoder(
                categorical_dims,
                encoding_methods,
                encoding_configs,
                true_categorical_columns,
                self.tr_loader,
            )

        self._categorical_dim = 0 if self.encoder is None else self.encoder.merged_dim
        self._numerical_columns = sorted(self.numerical_columns_mapping.values())

    @property
    @abstractmethod
    def input_sample(self) -> tensor_dict_type:
        x = self.tr_data.processed.x[:2]
        y = self.tr_data.processed.y[:2]
        x, y = map(to_torch, [x, y])
        return {"x_batch": x, "y_batch": y}

    @abstractmethod
    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # batch will have `categorical`, `numerical` and `labels` keys
        # requires returning `predictions` key
        pass

    @property
    def output_probabilities(self) -> bool:
        return False

    @property
    def use_ema(self) -> bool:
        return self.ema is not None

    def init_ema(self) -> None:
        ema_decay = self.config.setdefault("ema_decay", 0.0)
        if 0.0 < ema_decay < 1.0:
            named_params = list(self.named_parameters())
            self.ema = EMA(ema_decay, named_params)  # type: ignore

    def apply_ema(self) -> None:
        if self.ema is None:
            raise ValueError("`ema` is not defined")
        self.ema()

    def info(self, *, return_only: bool = False) -> str:
        msg = "\n".join(["=" * 100, "configurations", "-" * 100, ""])
        msg += (
            pprint.pformat(self._pipeline_config, compact=True)
            + "\n"
            + "-" * 100
            + "\n"
        )
        msg += "\n".join(["=" * 100, "parameters", "-" * 100, ""])
        for name, param in self.named_parameters():
            if param.requires_grad:
                msg += name + "\n"
        msg += "\n".join(["-" * 100, "=" * 100, "buffers", "-" * 100, ""])
        for name, param in self.named_buffers():
            msg += name + "\n"
        msg += "\n".join(
            ["-" * 100, "=" * 100, "structure", "-" * 100, str(self), "-" * 100, ""]
        )
        if not return_only:
            self.log_block_msg(msg, verbose_level=4)  # type: ignore
        all_msg, msg = msg, "=" * 100 + "\n"
        n_tr = len(self.tr_data)
        n_cv = None if self.cv_data is None else len(self.cv_data)
        msg += f"{self.info_prefix}training data : {n_tr}\n"
        msg += f"{self.info_prefix}valid    data : {n_cv}\n"
        msg += "-" * 100
        if not return_only:
            self.log_block_msg(msg, verbose_level=3)  # type: ignore
        return "\n".join([all_msg, msg])

    def loss_function(
        self,
        batch: tensor_dict_type,
        batch_indices: np.ndarray,
        forward_results: tensor_dict_type,
    ) -> Dict[str, torch.Tensor]:
        # requires returning `loss` key
        y_batch = batch["y_batch"]
        if self.tr_data.is_clf:
            y_batch = y_batch.view(-1)
        predictions = forward_results["predictions"]
        # `sample_weights` could be accessed through:
        # 1) `self.tr_weights[batch_indices]` (for training)
        # 2) `self.cv_weights[batch_indices]` (for validation)
        losses = self.loss(predictions, y_batch)
        return {"loss": losses.mean()}

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
        dims = {}
        if self.encoder is None:
            return dims
        merged_dims = self.encoder.merged_dims
        for idx in sorted(merged_dims):
            true_idx = self.categorical_columns_mapping[idx]
            dims[true_idx] = merged_dims[idx]
        return dims

    def _preset_config(self) -> None:
        pass

    def _init_config(self) -> None:
        self._loss_config = self.config.setdefault("loss_config", {})
        encoding_methods = self.config.setdefault("encoding_methods", {})
        encoding_configs = self.config.setdefault("encoding_configs", {})
        self._encoding_methods = {str(k): v for k, v in encoding_methods.items()}
        self._encoding_configs = {str(k): v for k, v in encoding_configs.items()}
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

    def _init_loss(self) -> None:
        if self.tr_data.is_reg:
            self.loss: nn.Module = nn.L1Loss(reduction="none")
        else:
            self.loss = FocalLoss(self._loss_config, reduction="none")

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
        batch_indices: Optional[np.ndarray],
    ) -> SplitFeatures:
        if self.encoder is None:
            return SplitFeatures(None, x_batch)
        with timing_context(self, "encoding"):
            encoding_result = self.encoder(x_batch, batch_indices)
        with timing_context(self, "fetch_numerical"):
            numerical = (
                None
                if not self._numerical_columns
                else x_batch[..., self._numerical_columns]
            )
        return SplitFeatures(encoding_result, numerical)

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device), None)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global model_dict

        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, model_dict, before_register=before)


__all__ = [
    "SplitFeatures",
    "ModelBase",
]
