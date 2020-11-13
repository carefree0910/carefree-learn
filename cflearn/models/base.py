import torch
import pprint

import numpy as np
import torch.nn as nn

from typing import *
from abc import abstractmethod
from abc import ABCMeta
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict
from torch.optim import Optimizer
from cfdata.tabular import ColumnTypes
from cfdata.tabular import DataLoader
from cftool.misc import register_core
from cftool.misc import timing_context
from cftool.misc import LoggingMixin

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
    numerical: Optional[Tensor]

    def merge(
        self,
        only_categorical: bool = False,
        use_embedding: bool = True,
        use_one_hot: bool = True,
    ) -> Tensor:
        if use_embedding and use_one_hot:
            return self._merge_all(only_categorical)
        numerical = None if only_categorical else self.numerical
        if not use_embedding and not use_one_hot:
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


class Transform(Module):
    def __init__(
        self,
        *,
        one_hot: bool,
        embedding: bool,
        only_categorical: bool,
    ):
        super().__init__()
        self.only_categorical = only_categorical
        self.use_embedding = embedding
        self.use_one_hot = one_hot

    def forward(self, split: SplitFeatures) -> Tensor:
        return split.merge(self.only_categorical, self.use_embedding, self.use_one_hot)

    def extra_repr(self) -> str:
        only_str = "" if not self.only_categorical else "(only): categorical\n"
        embedding_str = f"(use_embedding): {self.use_embedding}"
        one_hot_str = f"(use_one_hot): {self.use_one_hot}"
        return f"{only_str}{embedding_str}\n{one_hot_str}"


class ModelBase(Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        pipeline_config: Dict[str, Any],
        tr_loader: DataLoader,
        cv_loader: DataLoader,
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
        self.timing = self._pipeline_config["use_timing_context"]
        self.config = pipeline_config.setdefault("model_config", {})
        self.tr_loader = tr_loader
        self.cv_loader = cv_loader
        self.tr_data = tr_loader.data
        self.cv_data = None if cv_loader is None else cv_loader.data
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
            loaders = {"tr": self.tr_loader}
            if self.cv_loader is not None:
                loaders["cv"] = self.cv_loader
            encoder_config = self.config.setdefault("encoder_config", {})
            self.encoder = Encoder(
                encoder_config,
                categorical_dims,
                encoding_methods,
                encoding_configs,
                true_categorical_columns,
                loaders,
            )

        self._categorical_dim = 0 if self.encoder is None else self.encoder.merged_dim
        self._numerical_columns = sorted(self.numerical_columns_mapping.values())

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
        dims: Dict[int, int] = {}
        if self.encoder is None:
            return dims
        merged_dims = self.encoder.merged_dims
        for idx in sorted(merged_dims):
            true_idx = self.categorical_columns_mapping[idx]
            dims[true_idx] = merged_dims[idx]
        return dims

    # Inheritance

    @property
    def input_sample(self) -> tensor_dict_type:
        x = self.tr_data.processed.x[:2]
        y = self.tr_data.processed.y[:2]
        x, y = map(to_torch, [x, y])
        return {"x_batch": x, "y_batch": y}

    def define_transforms(self) -> None:
        self.add_transform("basic", True, True, False)

    def add_transform(
        self,
        key: str,
        one_hot: bool,
        embedding: bool,
        only_categorical: bool,
    ) -> None:
        self.transforms[key] = Transform(
            one_hot=one_hot,
            embedding=embedding,
            only_categorical=only_categorical,
        )


    @abstractmethod
    def forward(
        self,
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
        batch_step: int = 0,
        **kwargs: Any,
    ) -> tensor_dict_type:
        # batch will have `categorical`, `numerical` and `labels` keys
        # requires returning `predictions` key
        pass

    @property
    def output_probabilities(self) -> bool:
        return False

    # API

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
        batch_step: int,
    ) -> tensor_dict_type:
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

    def _split_features(
        self,
        x_batch: Tensor,
        batch_indices: Optional[np.ndarray],
        loader_name: Optional[str],
    ) -> SplitFeatures:
        if self.encoder is None:
            return SplitFeatures(None, x_batch)
        with timing_context(self, "encoding", enable=self.timing):
            encoding_result = self.encoder(x_batch, batch_indices, loader_name)
        with timing_context(self, "fetch_numerical", enable=self.timing):
            numerical = (
                None
                if not self._numerical_columns
                else x_batch[..., self._numerical_columns]
            )
        return SplitFeatures(encoding_result, numerical)

    @staticmethod
    def common_forward(
        instance: "ModelBase",
        batch: tensor_dict_type,
        batch_indices: Optional[np.ndarray] = None,
        loader_name: Optional[str] = None,
    ) -> tensor_dict_type:
        x_batch = batch["x_batch"]
        split = instance._split_features(x_batch, batch_indices, loader_name)
        net = instance.transforms["basic"](split)
        if instance.tr_data.is_ts:
            net = net.view(x_batch.shape[0], -1)
        net = instance.core(net)  # type: ignore
        return {"predictions": net}

    @staticmethod
    def get_core_config(instance: "ModelBase") -> Dict[str, Any]:
        in_dim: int = instance.config.get("in_dim")
        out_dim: int = instance.config.get("out_dim")
        default_out_dim = max(instance.tr_data.num_classes, 1)
        if in_dim is None:
            in_dim = instance.merged_dim
        if out_dim is None:
            out_dim = default_out_dim
        return {"in_dim": in_dim, "out_dim": out_dim}

    def _preset_config(self) -> None:
        pass

    def _init_config(self) -> None:
        # transform
        self.transforms = ModuleDict()
        self.define_transforms()
        # encoding
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
        # loss
        self._loss_config = self.config.setdefault("loss_config", {})

    def _init_loss(self) -> None:
        if self.tr_data.is_reg:
            self.loss: Module = nn.L1Loss(reduction="none")
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

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device), None, None)

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
