import os
import json
import torch
import pprint
import shutil
import inspect
import logging

import numpy as np
import torch.nn as nn

from typing import *
from cftool.ml import *
from cftool.misc import *
from cfdata.tabular import *
from tqdm import tqdm
from functools import partial
from trains import Task, Logger
from abc import ABCMeta, abstractmethod

try:
    import torch.cuda.amp as amp
except:
    amp = None

from .losses import *
from .modules import *
from .misc.toolkit import *

trains_logger: Union[Logger, None] = None
model_dict: Dict[str, Type["ModelBase"]] = {}


class amp_autocast_context(context_error_handler):
    def __init__(self, use_amp: bool):
        self._autocast = None if not use_amp else amp.autocast()

    def __enter__(self):
        if self._autocast is not None:
            self._autocast.__enter__()

    def _normal_exit(self, exc_type, exc_val, exc_tb):
        if self._autocast is not None:
            self._autocast.__exit__(exc_type, exc_val, exc_tb)


class SplitFeatures(NamedTuple):
    categorical: Union[torch.Tensor, tensor_dict_type, None]
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
    def __init__(
        self,
        config: Dict[str, Any],
        tr_data: TabularData,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self._wrapper_config = config
        self.config = config.setdefault("model_config", {})
        self._preset_config(tr_data)
        self._init_config(tr_data)
        self._init_loss(tr_data)
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
    def forward(self, batch: tensor_dict_type, **kwargs) -> tensor_dict_type:
        # batch will have `categorical`, `numerical` and `labels` keys
        # requires returning `predictions` key
        pass

    def loss_function(
        self,
        batch: tensor_dict_type,
        forward_results: tensor_dict_type,
    ) -> Dict[str, Union[torch.Tensor, float]]:
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

    def _preset_config(self, tr_data: TabularData):
        pass

    def _init_config(self, tr_data: TabularData):
        self.tr_data = tr_data
        self._loss_config = self.config.setdefault("loss_config", {})
        # TODO : optimize encodings by pre-calculate one-hot encodings in Wrapper
        self._encoding_methods = self.config.setdefault("encoding_methods", {})
        self._encoding_configs = self.config.setdefault("encoding_configs", {})
        self._default_encoding_configs = self.config.setdefault(
            "default_encoding_configs", {}
        )
        self._default_encoding_method = self.config.setdefault(
            "default_encoding_method", "embedding"
        )

    def _init_loss(self, tr_data: TabularData):
        if tr_data.is_reg:
            self.loss = nn.L1Loss(reduction="none")
        else:
            self.loss = FocalLoss(self._loss_config, reduction="none")

    def _init_encoder(self, idx: int):
        methods = self._encoding_methods.setdefault(idx, self._default_encoding_method)
        config = self._encoding_configs.setdefault(idx, self._default_encoding_configs)
        num_values = self.tr_data.recognizers[idx].num_unique_values
        if isinstance(methods, str):
            methods = [methods]
        encoders = [encoder_dict[method](idx, num_values, config) for method in methods]
        self.encoders[str(idx)] = EncoderStack(*encoders)

    def _optimizer_step(self, optimizers, grad_scalar):
        for opt in optimizers.values():
            if grad_scalar is None:
                opt.step()
            else:
                grad_scalar.step(opt)
                grad_scalar.update()
            opt.zero_grad()

    @staticmethod
    def _switch_requires_grad(params: List[torch.nn.Parameter], requires_grad: bool):
        for param in params:
            param.requires_grad_(requires_grad)

    @staticmethod
    def _collate_tensor_dicts(
        ds: List[tensor_dict_type],
        dim: int = 0,
    ) -> tensor_dict_type:
        results = {}
        d0 = ds[0]
        for k in d0.keys():
            if not isinstance(d0[k], torch.Tensor):
                continue
            tensors = []
            for rs in ds:
                tensor = rs[k]
                if len(tensor.shape) == 0:
                    tensor = tensor.reshape([1])
                tensors.append(tensor)
            results[k] = torch.cat(tensors, dim=dim)
        return results

    @staticmethod
    def to_prob(raw: np.ndarray) -> np.ndarray:
        return nn.functional.softmax(torch.from_numpy(raw), dim=1).numpy()

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
        if not categorical_columns:
            categorical = None
        elif not return_all_encodings:
            categorical = torch.cat(categorical_columns, dim=1)
        else:
            categorical = self._collate_tensor_dicts(categorical_columns, dim=1)
        numerical = (
            None
            if not self._numerical_columns
            else x_batch[..., self._numerical_columns]
        )
        return SplitFeatures(categorical, numerical)

    def get_split(self, processed: np.ndarray, device: torch.device) -> SplitFeatures:
        return self._split_features(torch.from_numpy(processed).to(device))

    @classmethod
    def register(cls, name: str):
        global model_dict

        def before(cls_):
            cls_.__identifier__ = name

        return register_core(name, model_dict, before_register=before)


class Pipeline(nn.Module, LoggingMixin):
    def __init__(
        self,
        model: ModelBase,
        tracker: Tracker,
        config: Dict[str, Any],
        verbose_level: int,
        is_loading: bool,
    ):
        super().__init__()
        self.tracker = tracker
        self._init_config(config, is_loading)
        self.model = model
        self._verbose_level = verbose_level
        self._no_grad_in_predict = True

    def _init_config(self, config, is_loading):
        self._wrapper_config = config
        self.config = config.setdefault("pipeline_config", {})
        self.batch_size = self.config.setdefault("batch_size", 128)
        self.cv_batch_size = self.config.setdefault(
            "cv_batch_size", 5 * self.batch_size
        )
        self.use_tqdm = self.config.setdefault("use_tqdm", True)
        self.min_epoch = int(self.config.setdefault("min_epoch", 0))
        self.num_epoch = int(
            self.config.setdefault("num_epoch", max(40, self.min_epoch))
        )
        self.max_epoch = int(
            self.config.setdefault("max_epoch", max(200, self.num_epoch))
        )
        self.max_snapshot_num = int(self.config.setdefault("max_snapshot_num", 5))
        self.snapshot_start_step = int(self.config.setdefault("snapshot_start_step", 0))
        self._num_step_per_snapshot = int(
            self.config.setdefault("num_step_per_snapshot", 0)
        )
        self.num_snapshot_per_epoch = int(
            self.config.setdefault("num_snapshot_per_epoch", 2)
        )
        self.max_step_per_snapshot = int(
            self.config.setdefault("max_step_per_snapshot", 1000)
        )
        self.plateau_monitor_start_snapshot = int(
            self.config.setdefault("plateau_monitor_start_snapshot", self.num_epoch)
        )

        self._clip_norm = self.config.setdefault("clip_norm", 0.0)
        ema_decay = self.config.setdefault("ema_decay", 0.0)
        if not 0.0 < ema_decay < 1.0:
            self.ema_decay = None
        else:
            self.ema_decay = EMA(ema_decay, self.model.named_parameters())

        self._use_amp = config["use_amp"]
        self.scaler = None if amp is None or not self._use_amp else amp.GradScaler()

        self._logging_path_ = config["_logging_path_"]
        self.logging_folder = config["logging_folder"]
        self.checkpoint_folder = self.config.setdefault(
            "checkpoint_folder", os.path.join(self.logging_folder, "checkpoints")
        )
        if not is_loading and os.path.isdir(self.checkpoint_folder):
            self.log_msg(
                f"'{self.checkpoint_folder}' already exists, all of its contents will be removed",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            shutil.rmtree(self.checkpoint_folder)

    def _init_data(self, tr_data, cv_data):
        self.tr_data, self.cv_data = tr_data, cv_data
        tr_sampler = ImbalancedSampler(tr_data, verbose_level=self._verbose_level)
        self.tr_loader = DataLoader(
            self.batch_size,
            tr_sampler,
            return_indices=True,
            verbose_level=self._verbose_level,
        )
        if cv_data is None:
            self.cv_loader = None
        else:
            cv_sampler = ImbalancedSampler(
                cv_data, shuffle=False, verbose_level=self._verbose_level
            )
            self.cv_loader = DataLoader(
                self.cv_batch_size,
                cv_sampler,
                return_indices=True,
                verbose_level=self._verbose_level,
            )

    def _define_optimizer(self, params_name, optimizer_base, optimizer_config):
        if params_name == "all":
            parameters = self.model.parameters()
        else:
            parameters = getattr(self.model, params_name)
        opt = self.optimizers[params_name] = optimizer_base(
            parameters, **optimizer_config
        )
        return opt

    def _init_optimizers(self):
        optimizers = self.config.setdefault("optimizers", {"all": {}})
        self.optimizers, self.schedulers = {}, {}
        for params_name, params_optimizer_config in optimizers.items():
            optimizer = params_optimizer_config.setdefault("optimizer", "adam")
            optimizer_config = params_optimizer_config.setdefault(
                "optimizer_config", {}
            )
            scheduler = params_optimizer_config.setdefault("scheduler", "plateau")
            scheduler_config = params_optimizer_config.setdefault(
                "scheduler_config", {}
            )
            # optimizer
            optimizer_config.setdefault("lr", 1e-3)
            if optimizer == "nag":
                optimizer_config.setdefault("momentum", 0.999)
                optimizer_config.setdefault("weight_decay", 1e-7)
            if scheduler == "warmup":
                multiplier = scheduler_config.setdefault("multiplier", 3)
                default_warm_up_step = min(
                    10 * len(self.tr_loader),
                    int(
                        0.25
                        * self.plateau_monitor_start_snapshot
                        * self.num_step_per_snapshot
                    ),
                )
                warmup_step = scheduler_config.setdefault(
                    "warmup_step", default_warm_up_step
                )
                self.plateau_monitor_start_snapshot += int(
                    warmup_step / self.num_step_per_snapshot
                )
                self.snapshot_start_step += warmup_step
                optimizer_config["lr"] /= multiplier
            optimizer_base = (
                optimizer_dict[optimizer] if isinstance(optimizer, str) else optimizer
            )
            opt = self._define_optimizer(params_name, optimizer_base, optimizer_config)
            self.config["optimizer_config"] = optimizer_config
            self._optimizer_type = optimizer
            # scheduler
            plateau_default_config: Dict[str, Any] = {"mode": "max"}
            plateau_default_config.setdefault("verbose", self._verbose_level_ >= 3)
            plateau_default_config.setdefault(
                "patience",
                max(10, self.snapshot_start_step // self.num_step_per_snapshot),
            )
            if scheduler == "plateau":
                scheduler_config = update_dict(scheduler_config, plateau_default_config)
            elif scheduler == "warmup":
                scheduler_afterwards_base = scheduler_config.get(
                    "scheduler_afterwards_base", "plateau"
                )
                scheduler_afterwards_config = scheduler_config.get(
                    "scheduler_afterwards_config", {}
                )
                if scheduler_afterwards_base is not None and isinstance(
                    scheduler_afterwards_base, str
                ):
                    if scheduler_afterwards_base == "plateau":
                        scheduler_afterwards_config = update_dict(
                            scheduler_afterwards_config, plateau_default_config
                        )
                    scheduler_afterwards_base = scheduler_dict[
                        scheduler_afterwards_base
                    ]
                scheduler_config[
                    "scheduler_afterwards_base"
                ] = scheduler_afterwards_base
                scheduler_config[
                    "scheduler_afterwards_config"
                ] = scheduler_afterwards_config
            if scheduler is None:
                self.schedulers[params_name] = None
            else:
                if isinstance(scheduler, str):
                    scheduler = scheduler_dict[scheduler]
                self.schedulers[params_name] = scheduler(opt, **scheduler_config)
        self.schedulers_requires_metric = set()
        for key, scheduler in self.schedulers.items():
            signature = inspect.signature(scheduler.step)
            for name, param in signature.parameters.items():
                if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if name == "metrics":
                        self.schedulers_requires_metric.add(key)

    def _init_metrics(self):
        # metrics
        metric_config = self.config.setdefault("metric_config", {})
        metric_types = metric_config.setdefault("types", "auto")
        if metric_types == "auto":
            if self.tr_data.is_reg:
                metric_types = ["mae", "mse"]
            else:
                metric_types = ["auc", "acc"]
        if not isinstance(metric_types, (list, tuple)):
            metric_types = [metric_types]
        self._metrics_need_loss = False
        self.metrics: Dict[str, Union[Metrics, None]] = {}
        self.metrics_decay: Dict[str, ScalarEMA] = {}
        metric_decay = metric_config.setdefault("decay", 0.1)
        for i, metric_type in enumerate(metric_types):
            if metric_type not in Metrics.sign_dict:
                # here, sub_metric must be one of the `loss_dict` keys
                self._metrics_need_loss = True
                self.metrics[metric_type] = None
            else:
                sub_metric_config = metric_config.setdefault(
                    f"{metric_type}_config", {}
                )
                self.metrics[metric_type] = Metrics(metric_type, sub_metric_config)
            if not 0.0 < metric_decay < 1.0:
                self.metrics_decay = None
            else:
                self.metrics_decay[metric_type] = ScalarEMA(metric_decay)
        self.metrics_weights = metric_config.setdefault("weights", {})
        for metric_type in metric_types:
            self.metrics_weights.setdefault(metric_type, 1.0)

    @property
    def start_snapshot(self):
        return (
            self._step_count >= self.snapshot_start_step
            and self._epoch_count > self.min_epoch
        )

    @property
    def start_monitor_plateau(self):
        return (
            self._step_count
            >= self.plateau_monitor_start_snapshot * self.num_step_per_snapshot
        )

    @property
    def num_step_per_snapshot(self):
        if self._num_step_per_snapshot > 0:
            return self._num_step_per_snapshot
        return max(
            1,
            min(
                self.max_step_per_snapshot,
                int(len(self.tr_loader) / self.num_snapshot_per_epoch),
            ),
        )

    # core

    def _to_tqdm(self, loader: DataLoader) -> Union[tqdm, DataLoader]:
        if not self.use_tqdm:
            return loader
        return tqdm(loader, total=len(loader), leave=False, position=2)

    def _collect_info(self, *, return_only: bool):
        msg = "\n".join(["=" * 100, "configurations", "-" * 100, ""])
        msg += (
            pprint.pformat(self._wrapper_config, compact=True) + "\n" + "-" * 100 + "\n"
        )
        msg += "\n".join(["=" * 100, "parameters", "-" * 100, ""])
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                msg += name + "\n"
        msg += "\n".join(["-" * 100, "=" * 100, "buffers", "-" * 100, ""])
        for name, param in self.named_buffers():
            msg += name + "\n"
        msg += "\n".join(
            ["-" * 100, "=" * 100, "structure", "-" * 100, str(self), "-" * 100, ""]
        )
        if not return_only:
            self.log_block_msg(msg, verbose_level=4)
        all_msg, msg = msg, "=" * 100 + "\n"
        n_tr = len(self.tr_data)
        n_cv = None if self.cv_data is None else len(self.cv_data)
        msg += f"{self.info_prefix}training data : {n_tr}\n"
        msg += f"{self.info_prefix}valid    data : {n_cv}\n"
        msg += "-" * 100
        if not return_only:
            self.log_block_msg(msg, verbose_level=3)
        return "\n".join([all_msg, msg])

    def _to_device(self, arr: Union[np.ndarray, None]) -> Union[torch.Tensor, None]:
        if arr is None:
            return arr
        return to_torch(arr).to(self.model.device)

    def _collate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> tensor_dict_type:
        tensors = list(map(self._to_device, [x_batch, y_batch]))
        if y_batch is not None and self.tr_data.is_clf:
            tensors[-1] = tensors[-1].to(torch.int64)
        return dict(zip(["x_batch", "y_batch"], tensors))

    def _clip_norm_step(self):
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self._clip_norm
        )

    def _optimizer_step(self):
        self.model._optimizer_step(self.optimizers, self.scaler)

    def _monitor_step(self):
        if self._step_count % self.num_step_per_snapshot == 0:
            score, metrics = self._get_metrics()
            if self.start_monitor_plateau:
                if not self._monitor.plateau_flag:
                    self.log_msg("start monitoring plateau", self.info_prefix, 2)
                self._monitor.plateau_flag = True
            if self.start_snapshot:
                if self._monitor.check_terminate(score):
                    return True
                for key, scheduler in self.schedulers.items():
                    if scheduler is not None:
                        kwargs = {}
                        if key in self.schedulers_requires_metric:
                            kwargs["metrics"] = score
                        scheduler.step(**kwargs)
        return False

    def _get_metrics(self) -> Tuple[float, Dict[str, float]]:
        tr_loader, cv_loader = self.tr_loader, self.cv_loader
        if cv_loader is None and self.tr_loader._n_siamese > 1:
            raise ValueError("cv set should be provided when n_siamese > 1")
        if cv_loader is not None:
            loader = cv_loader
        else:
            loader = tr_loader.copy()
            loader.return_indices = tr_loader.return_indices
            loader.enabled_sampling = False
        if not self._metrics_need_loss:
            losses = None
            results = self._predict(loader=loader)
            predictions, labels = map(results.get, ["predictions", "labels"])
        else:
            predictions = None
            loader = self._to_tqdm(loader)
            forward_dicts, loss_dicts, labels = [], [], []
            for (x_batch, y_batch), _ in loader:
                labels.append(y_batch)
                batch = self._collate_batch(x_batch, y_batch)
                with eval_context(self):
                    forward_dicts.append(self.model(batch))
                    loss_dicts.append(
                        self.model.loss_function(batch, forward_dicts[-1])
                    )
            losses, forwards = map(
                self.model._collate_tensor_dicts, [loss_dicts, forward_dicts]
            )
            losses = {k: v.mean().item() for k, v in losses.items()}
        use_decayed = False
        signs, metrics, decayed_metrics = {}, {}, {}
        for metric_type, metric_ins in self.metrics.items():
            if metric_ins is None:
                signs[metric_type] = -1
                sub_metric = metrics[metric_type] = losses[metric_type]
            else:
                signs[metric_type] = metric_ins.sign
                if self.tr_data.is_reg:
                    metric_predictions = predictions
                else:
                    if metric_ins.requires_prob:
                        metric_predictions = self.model.to_prob(predictions)
                    else:
                        metric_predictions = predictions.argmax(1).reshape([-1, 1])
                sub_metric = metrics[metric_type] = metric_ins.metric(
                    labels, metric_predictions
                )
            if self.metrics_decay is not None and self.start_snapshot:
                use_decayed = True
                decayed_metrics[metric_type] = self.metrics_decay[metric_type].update(
                    "metric", sub_metric
                )
        metrics_for_scoring = decayed_metrics if use_decayed else metrics
        if self.tracker is not None or trains_logger is not None:
            for name, value in metrics_for_scoring.items():
                if self.tracker is not None:
                    self.tracker.track_scalar(name, value, iteration=self._step_count)
                if trains_logger is not None:
                    trains_logger.report_scalar(
                        title="Evaluating",
                        series=name,
                        value=value,
                        iteration=self._step_count,
                    )
        weighted_scores = [
            v * signs[k] * self.metrics_weights[k]
            for k, v in metrics_for_scoring.items()
        ]
        score = sum(weighted_scores) / len(weighted_scores)

        if self._epoch_tqdm is not None:
            self._epoch_tqdm.set_postfix(metrics_for_scoring)

        def _metric_verbose(k):
            metric_str = fix_float_to_length(metrics[k], 8)
            if not use_decayed:
                return metric_str
            return f"{metric_str} (ema: {fix_float_to_length(decayed_metrics[k], 8)})"

        msg = (
            f"| epoch {self._epoch_count:^4d} - step {self._step_count:^6d} | "
            f"{' | '.join([f'{k} : {_metric_verbose(k)}' for k in sorted(metrics)])} | "
            f"score : {fix_float_to_length(score, 8)} |"
        )
        with open(self._log_file, "a") as f:
            f.write(f"{msg}\n")
        self.log_msg(msg, verbose_level=None)

        return score, metrics

    def _get_results(
        self,
        no_grad: bool,
        loader: DataLoader,
        **kwargs,
    ) -> Tuple[List[np.ndarray], List[tensor_dict_type]]:
        return_indices, loader = loader.return_indices, self._to_tqdm(loader)
        with eval_context(self, no_grad=no_grad):
            results, labels = [], []
            for a, b in loader:
                if return_indices:
                    x_batch, y_batch = a
                else:
                    x_batch, y_batch = a, b
                if y_batch is not None:
                    labels.append(y_batch)
                results.append(
                    self.model(self._collate_batch(x_batch, y_batch), **kwargs)
                )
        return labels, results

    def _predict(self, loader: DataLoader, **kwargs) -> Dict[str, np.ndarray]:
        no_grad = kwargs.pop("no_grad", self._no_grad_in_predict)
        try:
            labels, results = self._get_results(no_grad, loader, **kwargs)
        except:
            no_grad = self._no_grad_in_predict = False
            labels, results = self._get_results(no_grad, loader, **kwargs)
        results = self.model._collate_tensor_dicts(results)
        results = {k: to_numpy(v) for k, v in results.items()}
        if labels:
            labels = np.vstack(labels)
            results["labels"] = labels
        return results

    # api

    def forward(
        self,
        tr_data: TabularData,
        cv_data: TabularData,
        tr_weights: np.ndarray,
    ):
        # data
        self._init_data(tr_data, cv_data)
        # sample weights
        if tr_weights is not None:
            tr_weights = to_torch(tr_weights)
        # optimizer
        self._init_optimizers()
        # metrics
        self._init_metrics()
        # monitor
        self._monitor = TrainMonitor(1).register_pipeline(self)
        # train
        self._collect_info(return_only=False)
        terminate = False
        self._step_count = self._epoch_count = 0
        tuple(
            map(
                lambda n: os.makedirs(n, exist_ok=True),
                [self.logging_folder, self.checkpoint_folder],
            )
        )
        log_name = f"{timestamp()}.txt"
        self._log_file = os.path.join(self.logging_folder, log_name)
        with open(self._log_file, "w"):
            pass
        self._step_tqdm = self._epoch_tqdm = None
        if self.use_tqdm:
            self._epoch_tqdm = tqdm(list(range(self.num_epoch)), position=0)
        while self._epoch_count < self.num_epoch:
            try:
                self._epoch_count += 1
                self._step_tqdm = iter(self.tr_loader)
                if self.use_tqdm:
                    self._step_tqdm = tqdm(
                        self._step_tqdm,
                        total=len(self.tr_loader),
                        position=1,
                        leave=False,
                    )
                for (x_batch, y_batch), index_batch in self._step_tqdm:
                    self._step_count += 1
                    with timing_context(self, "collate batch"):
                        batch = self._collate_batch(x_batch, y_batch)
                    with amp_autocast_context(self._use_amp):
                        with timing_context(self, "model.forward"):
                            forward_results = self.model(batch)
                        with timing_context(self, "loss.forward"):
                            if tr_weights is not None:
                                batch_sample_weights = tr_weights[index_batch]
                                forward_results[
                                    "batch_sample_weights"
                                ] = batch_sample_weights
                            loss_dict = self.model.loss_function(batch, forward_results)
                    if self.tracker is not None or trains_logger is not None:
                        for name, tensor in loss_dict.items():
                            value = tensor.item()
                            if self.tracker is not None:
                                self.tracker.track_scalar(
                                    name, value, iteration=self._step_count
                                )
                            if trains_logger is not None:
                                trains_logger.report_scalar(
                                    "Training",
                                    series=name,
                                    value=value,
                                    iteration=self._step_count,
                                )
                    with timing_context(self, "loss.backward"):
                        loss = loss_dict["loss"]
                        if self._use_amp:
                            loss = self.scaler.scale(loss)
                        loss.backward()
                    if self._clip_norm > 0.0:
                        with timing_context(self, "clip_norm_step"):
                            self._clip_norm_step()
                    with timing_context(self, "optimizer_step"):
                        self._optimizer_step()
                    if self.ema_decay is not None:
                        with timing_context(self, "EMA"):
                            self.ema_decay()
                    with timing_context(self, "monitor_step"):
                        terminate = self._monitor_step()
                    if terminate:
                        break
            except KeyboardInterrupt:
                self.log_msg(
                    "keyboard interrupted", self.error_prefix, msg_level=logging.ERROR
                )
                terminate = True
            if terminate:
                if os.path.isdir(self.checkpoint_folder):
                    self.log_msg(
                        "rolling back to the best checkpoint", self.info_prefix, 3
                    )
                    self.restore_checkpoint()
                break
            if self.use_tqdm:
                self._epoch_tqdm.total = self.num_epoch
                self._epoch_tqdm.update()
        if self.use_tqdm:
            if self._step_tqdm is not None:
                self._step_tqdm.close()
            self._epoch_tqdm.close()
        self._step_count = self._epoch_count = -1
        self._get_metrics()

    def predict(
        self,
        x: data_type,
        return_all: bool = False,
        *,
        contains_labels: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        data = self.tr_data.copy_to(x, None, contains_labels=contains_labels)
        loader = DataLoader(self.cv_batch_size, ImbalancedSampler(data, shuffle=False))
        predictions = self._predict(loader, **kwargs)
        if return_all:
            return predictions
        return predictions["predictions"]

    @staticmethod
    def _filter_checkpoints(folder) -> Dict[int, str]:
        checkpoints = {}
        for file in os.listdir(folder):
            if file.startswith("pipeline_") and file.endswith(".pt"):
                step = int(os.path.splitext(file)[0].split("_")[1])
                checkpoints[step] = file
        return checkpoints

    def save_checkpoint(self, folder=None):
        if folder is None:
            folder = self.checkpoint_folder
        if self.max_snapshot_num > 0:
            checkpoints = self._filter_checkpoints(folder)
            if len(checkpoints) >= self.max_snapshot_num - 1:
                for key in sorted(checkpoints)[: -self.max_snapshot_num + 1]:
                    os.remove(os.path.join(folder, checkpoints[key]))
        file = f"pipeline_{self._step_count}.pt"
        torch.save(self.state_dict(), os.path.join(folder, file))

    def restore_checkpoint(self, folder=None):
        if folder is None:
            folder = self.checkpoint_folder
        checkpoints = self._filter_checkpoints(folder)
        if not checkpoints:
            self.log_msg(
                f"no pipeline file found in {self.checkpoint_folder}",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            return self
        latest_checkpoint = checkpoints[sorted(checkpoints)[-1]]
        pipeline_file = os.path.join(folder, latest_checkpoint)
        self.log_msg(f"restoring from {pipeline_file}", self.info_prefix, 4)
        self.load_state_dict(torch.load(pipeline_file, map_location=self.model.device))
        return self


class Wrapper(LoggingMixin):
    def __init__(
        self,
        config: Union[str, Dict[str, Any]] = None,
        *,
        increment_config: Union[str, Dict[str, Any]] = None,
        tracker_config: Dict[str, Any] = None,
        cuda: Union[str, int] = None,
        verbose_level: int = 2,
    ):
        self.tracker = None if tracker_config is None else Tracker(**tracker_config)
        self._verbose_level = int(verbose_level)
        if cuda == "cpu":
            self.device = torch.device("cpu")
        elif cuda is not None:
            self.device = torch.device(f"cuda:{cuda}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config, increment_config = map(
            self._get_config, [config, increment_config]
        )
        update_dict(increment_config, self.config)
        self._init_config()

    def __str__(self):
        return f"{type(self.model).__name__}()"

    __repr__ = __str__

    @property
    def train_set(self) -> TabularDataset:
        raw = self.tr_data.raw
        return TabularDataset(*raw.xy, task_type=self.tr_data.task_type)

    @property
    def valid_set(self) -> Union[TabularDataset, None]:
        if self.cv_data is None:
            return
        raw = self.cv_data.raw
        return TabularDataset(*raw.xy, task_type=self.cv_data.task_type)

    @property
    def binary_threshold(self) -> Union[float, None]:
        return self._binary_threshold

    @staticmethod
    def _get_config(config):
        if config is None:
            return {}
        if isinstance(config, str):
            with open(config, "r") as f:
                return json.load(f)
        return shallow_copy_dict(config)

    def _init_config(self):
        self._data_config = self.config.setdefault("data_config", {})
        self._data_config["default_categorical_process"] = "identical"
        self._read_config = self.config.setdefault("read_config", {})
        self._cv_split = self.config.setdefault("cv_split", 0.1)
        self._model = self.config.setdefault("model", "fcnn")
        self._binary_metric = self.config.setdefault("binary_metric", "acc")
        self._is_binary = self.config.get("is_binary")
        self._binary_threshold = self.config.get("binary_threshold")
        self.config.setdefault("use_amp", False)
        logging_folder = self.config["logging_folder"] = self.config.setdefault(
            "logging_folder",
            os.path.join("_logging", model_dict[self._model].__identifier__),
        )
        logging_file = self.config.get("logging_file")
        if logging_file is not None:
            logging_path = os.path.join(logging_folder, logging_file)
        else:
            logging_path = os.path.abspath(self.generate_logging_path(logging_folder))
        self.config["_logging_path_"] = logging_path
        self._init_logging(
            self._verbose_level, self.config.setdefault("trigger_logging", False)
        )

    def _prepare_modules(self, *, is_loading: bool = False):
        # model
        with timing_context(self, "init model"):
            self.model = model_dict[self._model](self.config, self.tr_data, self.device)
        # pipeline
        with timing_context(self, "init pipeline"):
            self.pipeline = Pipeline(
                self.model, self.tracker, self.config, self._verbose_level, is_loading
            )
        # to device
        with timing_context(self, "init device"):
            self.pipeline.to(self.device)

    def _before_loop(
        self,
        x: data_type,
        y: data_type,
        x_cv: data_type,
        y_cv: data_type,
        sample_weights: np.ndarray,
    ):
        # data
        y, y_cv = map(to_2d, [y, y_cv])
        args = (x, y) if y is not None else (x,)
        self._data_config["verbose_level"] = self._verbose_level
        self._original_data = TabularData(**self._data_config).read(
            *args, **self._read_config
        )
        self.tr_data = self._original_data
        self._save_original_data = False
        self.tr_weights = None
        if x_cv is not None:
            self.cv_data = self.tr_data.copy_to(x_cv, y_cv)
            if sample_weights is not None:
                self.tr_weights = sample_weights[: len(self.tr_data)]
        else:
            if self._cv_split <= 0.0:
                self.cv_data = None
                if sample_weights is not None:
                    self.tr_weights = sample_weights
            else:
                self._save_original_data = True
                split = self.tr_data.split(self._cv_split)
                self.cv_data, self.tr_data = split.split, split.remained
                # TODO : utilize cv_weights with sample_weights[split.split_indices]
                if sample_weights is not None:
                    self.tr_weights = sample_weights[split.remained_indices]
        # modules
        self._prepare_modules()

    def _loop(self):
        # training loop
        self.pipeline(self.tr_data, self.cv_data, self.tr_weights)
        # binary threshold
        if self._binary_threshold is None:
            if self.tr_data.num_classes != 2:
                self._is_binary = False
                self._binary_threshold = None
            else:
                self._is_binary = True
                x, y = self.tr_data.raw.x, self.tr_data.processed.y
                probabilities = self.predict_prob(x)
                try:
                    threshold = Metrics.get_binary_threshold(
                        y, probabilities, self._binary_metric
                    )
                    self._binary_threshold = threshold
                except ValueError:
                    self._binary_threshold = None
        # logging
        self.log_timing()

    # api

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        sample_weights: np.ndarray = None,
    ) -> "Wrapper":
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self._loop()
        return self

    def trains(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        sample_weights: np.ndarray = None,
        trains_config: Dict[str, Any] = None,
        keep_task_open: bool = False,
        queue: str = None,
    ) -> "Wrapper":
        if trains_config is None:
            return self.fit(x, y, x_cv, y_cv, sample_weights=sample_weights)
        # init trains
        if trains_config is None:
            trains_config = {}
        project_name = trains_config.get("project_name")
        task_name = trains_config.get("task_name")
        if queue is None:
            task = Task.init(**trains_config)
            cloned_task = None
        else:
            task = Task.get_task(project_name=project_name, task_name=task_name)
            cloned_task = Task.clone(source_task=task, parent=task.id)
        # before loop
        self._verbose_level = 6
        self._data_config["verbose_level"] = 6
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self.pipeline.use_tqdm = False
        copied_config = shallow_copy_dict(self.config)
        if queue is not None:
            cloned_task.set_parameters(copied_config)
            Task.enqueue(cloned_task.id, queue)
            return self
        # loop
        task.connect(copied_config)
        global trains_logger
        trains_logger = task.get_logger()
        self._loop()
        if not keep_task_open:
            task.close()
            trains_logger = None
        return self

    def predict(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        requires_recover: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.tr_data.is_reg:
            predictions = self.pipeline.predict(x, return_all, **kwargs)
            recover = partial(self.tr_data.recover_labels, inplace=True)
            if not return_all:
                if requires_recover:
                    return recover(predictions)
                return predictions
            if not requires_recover:
                return predictions
            return {k: recover(v) for k, v in predictions.items()}
        probabilities = self.predict_prob(x, **kwargs)
        if not self._is_binary or self._binary_threshold is None:
            return probabilities.argmax(1).reshape([-1, 1])
        return (
            (probabilities[..., 1] >= self._binary_threshold)
            .astype(np.int)
            .reshape([-1, 1])
        )

    def predict_prob(self, x: data_type, **kwargs) -> np.ndarray:
        if self.tr_data.is_reg:
            raise ValueError("`predict_prob` should not be called on regression tasks")
        raw = self.pipeline.predict(x, **kwargs)
        return self.model.to_prob(raw)

    def to_pattern(
        self,
        *,
        pre_process: callable = None,
        **predict_kwargs,
    ) -> ModelPattern:
        def _predict(x):
            if pre_process is not None:
                x = pre_process(x)
            return self.predict(x, **predict_kwargs)

        def _predict_prob(x):
            if pre_process is not None:
                x = pre_process(x)
            return self.predict_prob(x, **predict_kwargs)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    def save(self, folder: str = None, *, compress: bool = True) -> "Wrapper":
        if folder is None:
            folder = self.pipeline.checkpoint_folder
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            Saving.prepare_folder(self, folder)
            train_data_folder = os.path.join(folder, "__data__", "train")
            if self._save_original_data:
                self._original_data.save(train_data_folder, compress=compress)
            else:
                self.tr_data.save(train_data_folder, compress=compress)
                if self.cv_data is not None:
                    self.cv_data.save(
                        os.path.join(folder, "__data__", "valid"), compress=compress
                    )
            self.pipeline.save_checkpoint(folder)
            self.config["is_binary"] = self._is_binary
            self.config["binary_threshold"] = self._binary_threshold
            Saving.save_dict(self.config, "config", folder)
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        folder: str,
        *,
        cuda: int = None,
        verbose_level: int = 0,
        compress: bool = True,
    ) -> "Wrapper":
        base_folder = os.path.dirname(os.path.abspath(folder))
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(folder, compress, remove_extracted=True):
                config = Saving.load_dict("config", folder)
                wrapper = Wrapper(config, cuda=cuda, verbose_level=verbose_level)
                tr_data_folder = os.path.join(folder, "__data__", "train")
                cv_data_folder = os.path.join(folder, "__data__", "valid")
                tr_data = wrapper.tr_data = TabularData.load(
                    tr_data_folder, compress=compress
                )
                cv_data = None
                if os.path.isdir(cv_data_folder) or os.path.isfile(
                    f"{cv_data_folder}.zip"
                ):
                    cv_data = wrapper.cv_data = TabularData.load(
                        cv_data_folder, compress=compress
                    )
                wrapper._prepare_modules(is_loading=True)
                pipeline = wrapper.pipeline
                pipeline.restore_checkpoint(folder)
                pipeline._init_data(tr_data, cv_data)
                pipeline._init_metrics()
        return wrapper


__all__ = ["ModelBase", "model_dict", "Pipeline", "Wrapper"]
