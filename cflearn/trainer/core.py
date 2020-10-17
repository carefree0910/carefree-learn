import os
import torch
import optuna
import pprint
import shutil
import inspect
import logging

import numpy as np
import torch.nn as nn

from typing import *
from cftool.ml import Metrics
from cftool.ml import Tracker
from cftool.ml import ScalarEMA
from cfdata.tabular import DataLoader
from cftool.misc import LoggingMixin
from cftool.misc import timestamp
from cftool.misc import update_dict
from cftool.misc import timing_context
from cftool.misc import fix_float_to_length
from cftool.misc import context_error_handler
from tqdm import tqdm
from trains import Logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from cfdata.types import np_int_type

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..misc.toolkit import *
from ..types import np_dict_type
from ..types import tensor_dict_type
from ..modules import optimizer_dict
from ..modules import scheduler_dict
from ..modules import EMA
from ..models.base import collate_np_dicts
from ..models.base import collate_tensor_dicts
from ..models.base import ModelBase


trains_logger: Union[Logger, None] = None


def to_prob(raw: np.ndarray) -> np.ndarray:
    return nn.functional.softmax(torch.from_numpy(raw), dim=1).numpy()


class amp_autocast_context(context_error_handler):
    def __init__(self, use_amp: bool):
        if not use_amp:
            self._autocast = None
        else:
            if amp is None:
                raise ValueError("`amp` is not available but `use_amp` is set to True")
            self._autocast = amp.autocast()

    def __enter__(self) -> None:
        if self._autocast is not None:
            self._autocast.__enter__()

    def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._autocast is not None:
            self._autocast.__exit__(exc_type, exc_val, exc_tb)


class Trainer(nn.Module, LoggingMixin):
    def __init__(
        self,
        model: ModelBase,
        trial: Optional[optuna.trial.Trial],
        tracker: Tracker,
        pipeline_config: Dict[str, Any],
        verbose_level: int,
        is_loading: bool,
    ):
        super().__init__()
        self.trial = trial
        self.tracker = tracker
        self._init_config(pipeline_config, is_loading)
        self.model = model
        self._verbose_level = verbose_level
        self._use_grad_in_predict = False
        self.onnx: Optional[Any] = None

    def _init_config(self, pipeline_config: Dict[str, Any], is_loading: bool) -> None:
        self._pipeline_config = pipeline_config
        self.timing = self._pipeline_config["use_timing_context"]
        self.config = pipeline_config.setdefault("trainer_config", {})
        self.use_tqdm = self.config.setdefault("use_tqdm", True)
        self.min_epoch = int(self.config.setdefault("min_epoch", 0))
        num_epoch = self.config.setdefault("num_epoch", max(40, self.min_epoch))
        max_epoch = self.config.setdefault("max_epoch", max(200, num_epoch))
        self.num_epoch, self.max_epoch = map(int, [num_epoch, max_epoch])
        self.max_snapshot_num = int(self.config.setdefault("max_snapshot_num", 5))
        self.snapshot_start_step = int(self.config.setdefault("snapshot_start_step", 0))
        num_step_per_snapshot = self.config.setdefault("num_step_per_snapshot", 0)
        num_snapshot_per_epoch = self.config.setdefault("num_snapshot_per_epoch", 2)
        max_step_per_snapshot = self.config.setdefault("max_step_per_snapshot", 1000)
        plateau_start = self.config.setdefault("plateau_start_snapshot", num_epoch)
        self._num_step_per_snapshot = int(num_step_per_snapshot)
        self.num_snapshot_per_epoch = int(num_snapshot_per_epoch)
        self.max_step_per_snapshot = int(max_step_per_snapshot)
        self.plateau_start = int(plateau_start)

        self._clip_norm = self.config.setdefault("clip_norm", 0.0)
        ema_decay = self.config.setdefault("ema_decay", 0.0)
        if not 0.0 < ema_decay < 1.0:
            self.ema_decay = None
        else:
            named_params = list(self.model.named_parameters())
            self.ema_decay = EMA(ema_decay, named_params)  # type: ignore

        self._use_amp = pipeline_config["use_amp"]
        self.scaler = None if amp is None or not self._use_amp else amp.GradScaler()

        self._logging_path_ = pipeline_config["_logging_path_"]
        self.logging_folder = pipeline_config["logging_folder"]
        default_checkpoint_folder = os.path.join(self.logging_folder, "checkpoints")
        self.checkpoint_folder = self.config.setdefault(
            "checkpoint_folder", default_checkpoint_folder
        )
        if not is_loading and os.path.isdir(self.checkpoint_folder):
            self.log_msg(  # type: ignore
                f"'{self.checkpoint_folder}' already exists, "
                "all of its contents will be removed",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            shutil.rmtree(self.checkpoint_folder)

    def _define_optimizer(
        self,
        params_name: str,
        optimizer_base: Type[Optimizer],
        optimizer_config: Dict[str, Any],
    ) -> Optimizer:
        if params_name == "all":
            parameters = self.model.parameters()
        else:
            parameters = getattr(self.model, params_name)
        opt = self.optimizers[params_name] = optimizer_base(
            parameters, **optimizer_config
        )
        return opt

    def _init_optimizers(self) -> None:
        optimizers_settings = self.config.setdefault("optimizers", {"all": {}})
        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, Optional[_LRScheduler]] = {}
        for params_name, opt_setting in optimizers_settings.items():
            optimizer = opt_setting.setdefault("optimizer", "adam")
            optimizer_config = opt_setting.setdefault("optimizer_config", {})
            scheduler = opt_setting.setdefault("scheduler", "plateau")
            scheduler_config = opt_setting.setdefault("scheduler_config", {})
            # optimizer
            optimizer_config.setdefault("lr", 1e-3)
            if optimizer == "nag":
                optimizer_config.setdefault("momentum", 0.999)
                optimizer_config.setdefault("weight_decay", 1e-7)
            if scheduler == "warmup":
                multiplier = scheduler_config.setdefault("multiplier", 3)
                default_warm_up_step = min(
                    10 * len(self.tr_loader),
                    int(0.25 * self.plateau_start * self.num_step_per_snapshot),
                )
                warmup_step = scheduler_config.setdefault(
                    "warmup_step", default_warm_up_step
                )
                self.plateau_start += int(warmup_step / self.num_step_per_snapshot)
                self.snapshot_start_step += warmup_step
                optimizer_config["lr"] /= multiplier
            optimizer_base = (
                optimizer_dict[optimizer] if isinstance(optimizer, str) else optimizer
            )
            opt = self._define_optimizer(params_name, optimizer_base, optimizer_config)
            self.config["optimizer_config"] = optimizer_config
            self._optimizer_type = optimizer
            # scheduler
            plateau_default_cfg: Dict[str, Any] = {"mode": "max"}
            assert isinstance(self._verbose_level_, int)
            plateau_default_cfg.setdefault("verbose", self._verbose_level_ >= 3)
            plateau_default_cfg.setdefault(
                "patience",
                max(10, self.snapshot_start_step // self.num_step_per_snapshot),
            )
            if scheduler == "plateau":
                scheduler_config = update_dict(scheduler_config, plateau_default_cfg)
            elif scheduler == "warmup":
                sab = scheduler_config.get("scheduler_afterwards_base", "plateau")
                sac = scheduler_config.get("scheduler_afterwards_config", {})
                if sab is not None and isinstance(sab, str):
                    if sab == "plateau":
                        sac = update_dict(sac, plateau_default_cfg)
                    sab = scheduler_dict[sab]
                scheduler_config["scheduler_afterwards_base"] = sab
                scheduler_config["scheduler_afterwards_config"] = sac
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

    def _init_metrics(self) -> None:
        # metrics
        metric_config = self.config.setdefault("metric_config", {})
        metric_types = metric_config.setdefault("types", "auto")
        if metric_types == "auto":
            if self.tr_loader.data.is_reg:
                metric_types = ["mae", "mse"]
            else:
                metric_types = ["auc", "acc"]
        if not isinstance(metric_types, (list, tuple)):
            metric_types = [metric_types]
        self._metrics_need_loss = False
        self.metrics: Dict[str, Union[Metrics, None]] = {}
        self.metrics_decay: Optional[Dict[str, ScalarEMA]] = {}
        metric_decay = metric_config.setdefault("decay", 0.1)
        for i, metric_type in enumerate(metric_types):
            if metric_type not in Metrics.sign_dict:
                # here, sub_metric must be one of the `loss_dict` keys
                self._metrics_need_loss = True
                self.metrics[metric_type] = None
            else:
                metric_key = f"{metric_type}_config"
                sub_metric_config = metric_config.setdefault(metric_key, {})
                self.metrics[metric_type] = Metrics(metric_type, sub_metric_config)
            if not 0.0 < metric_decay < 1.0:
                self.metrics_decay = None
            else:
                assert isinstance(self.metrics_decay, dict)
                self.metrics_decay[metric_type] = ScalarEMA(metric_decay)
        self.metrics_weights = metric_config.setdefault("weights", {})
        for metric_type in metric_types:
            self.metrics_weights.setdefault(metric_type, 1.0)

    @property
    def start_snapshot(self) -> bool:
        return (
            self._step_count >= self.snapshot_start_step
            and self._epoch_count > self.min_epoch
        )

    @property
    def start_monitor_plateau(self) -> bool:
        return self._step_count >= self.plateau_start * self.num_step_per_snapshot

    @property
    def num_step_per_snapshot(self) -> int:
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

    def _collect_info(self, *, return_only: bool) -> str:
        msg = "\n".join(["=" * 100, "configurations", "-" * 100, ""])
        msg += (
            pprint.pformat(self._pipeline_config, compact=True)
            + "\n"
            + "-" * 100
            + "\n"
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
            self.log_block_msg(msg, verbose_level=4)  # type: ignore
        all_msg, msg = msg, "=" * 100 + "\n"
        n_tr = len(self.tr_loader.sampler)
        n_cv = None if self.cv_loader is None else len(self.cv_loader.sampler)
        msg += f"{self.info_prefix}training data : {n_tr}\n"
        msg += f"{self.info_prefix}valid    data : {n_cv}\n"
        msg += "-" * 100
        if not return_only:
            self.log_block_msg(msg, verbose_level=3)  # type: ignore
        return "\n".join([all_msg, msg])

    def _to_device(self, arr: Union[np.ndarray, None]) -> Union[torch.Tensor, None]:
        if arr is None:
            return arr
        arr = to_torch(arr)
        if self.onnx is None:
            arr = arr.to(self.model.device)
        return arr

    def _collate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> tensor_dict_type:
        if self.onnx is not None:
            if y_batch is None:
                y_batch = np.zeros([*x_batch.shape[:-1], 1], np_int_type)
            tensors = [x_batch, y_batch]
        else:
            x_batch, y_batch = map(self._to_device, [x_batch, y_batch])
            if y_batch is not None and self.tr_loader.data.is_clf:
                y_batch = y_batch.to(torch.int64)
            tensors = [x_batch, y_batch]
        return dict(zip(["x_batch", "y_batch"], tensors))

    def _clip_norm_step(self) -> None:
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self._clip_norm
        )

    def _optimizer_step(self) -> None:
        self.model._optimizer_step(self.optimizers, self.scaler)

    # return whether we need to terminate
    def _monitor_step(self) -> bool:
        if self._step_count % self.num_step_per_snapshot == 0:
            score, metrics = self._get_metrics()
            if self.trial is not None:
                self.trial.report(score, step=self._step_count)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
            if self.start_monitor_plateau:
                if not self._monitor.plateau_flag:
                    self.log_msg(  # type: ignore
                        "start monitoring plateau",
                        self.info_prefix,
                        2,
                    )
                self._monitor.plateau_flag = True
            if self.start_snapshot:
                if self._monitor.check_terminate(score):
                    return True
                for key, scheduler in self.schedulers.items():
                    if scheduler is not None:
                        kwargs = {}
                        if key in self.schedulers_requires_metric:
                            kwargs["metrics"] = score
                        scheduler.step(**kwargs)  # type: ignore
        return False

    def _get_metrics(self) -> Tuple[float, Dict[str, float]]:
        tr_loader, cv_loader = self.tr_loader, self.cv_loader
        if cv_loader is None and self.tr_loader._num_siamese > 1:
            raise ValueError("cv set should be provided when num_siamese > 1")
        if cv_loader is not None:
            loader = cv_loader
        else:
            loader = tr_loader.copy()
            loader.return_indices = tr_loader.return_indices
            loader.enabled_sampling = False
        if not self._metrics_need_loss:
            loss_values = None
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
            losses, forwards = map(collate_tensor_dicts, [loss_dicts, forward_dicts])
            loss_values = {k: v.mean().item() for k, v in losses.items()}
        use_decayed = False
        signs: Dict[str, int] = {}
        metrics: Dict[str, float] = {}
        decayed_metrics: Dict[str, float] = {}
        for metric_type, metric_ins in self.metrics.items():
            if metric_ins is None:
                assert loss_values is not None
                signs[metric_type] = -1
                sub_metric = metrics[metric_type] = loss_values[metric_type]
            else:
                signs[metric_type] = metric_ins.sign
                if self.tr_loader.data.is_reg:
                    metric_predictions = predictions
                else:
                    if metric_ins.requires_prob:
                        metric_predictions = to_prob(predictions)
                    else:
                        assert isinstance(predictions, np.ndarray)
                        metric_predictions = predictions.argmax(1).reshape([-1, 1])
                sub_metric = metric_ins.metric(labels, metric_predictions)
                metrics[metric_type] = sub_metric
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
        score: float = sum(weighted_scores) / len(weighted_scores)

        if self._epoch_tqdm is not None:
            self._epoch_tqdm.set_postfix(metrics_for_scoring)

        def _metric_verbose(k: str) -> str:
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
        self.log_msg(msg, verbose_level=None)  # type: ignore

        return score, metrics

    def _get_results(
        self,
        use_grad: bool,
        loader: DataLoader,
        **kwargs: Any,
    ) -> Tuple[List[np.ndarray], List[np_dict_type]]:
        return_indices, loader = loader.return_indices, self._to_tqdm(loader)
        with eval_context(self, use_grad=use_grad):
            results, labels = [], []
            for a, b in loader:
                if return_indices:
                    x_batch, y_batch = a
                else:
                    x_batch, y_batch = a, b
                if y_batch is not None:
                    labels.append(y_batch)
                batch = self._collate_batch(x_batch, y_batch)
                if self.onnx is not None:
                    rs = self.onnx.inference(batch)
                else:
                    rs = self.model(batch, **kwargs)
                    for k, v in rs.items():
                        if isinstance(v, torch.Tensor):
                            rs[k] = to_numpy(v)
                results.append(rs)
        return labels, results

    def _predict(self, loader: DataLoader, **kwargs: Any) -> np_dict_type:
        use_grad = kwargs.pop("use_grad", self._use_grad_in_predict)
        try:
            labels, results = self._get_results(use_grad, loader, **kwargs)
        except:
            use_grad = self._use_grad_in_predict = True
            labels, results = self._get_results(use_grad, loader, **kwargs)
        collated = collate_np_dicts(results)
        if labels:
            labels = np.vstack(labels)
            collated["labels"] = labels
        return collated

    # api

    def forward(
        self,
        tr_loader: DataLoader,
        cv_loader: DataLoader,
        tr_weights: np.ndarray,
    ) -> None:
        self.tr_loader, self.cv_loader = tr_loader, cv_loader
        # sample weights
        if tr_weights is not None:
            tr_weights = to_torch(tr_weights)
        # optimizer
        self._init_optimizers()
        # metrics
        self._init_metrics()
        # monitor
        self._monitor = TrainMonitor(1).register_trainer(self)
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
        step_tqdm_legacy = self._epoch_tqdm = None
        if self.use_tqdm:
            self._epoch_tqdm = tqdm(list(range(self.num_epoch)), position=0)
        while self._epoch_count < self.num_epoch:
            try:
                self._epoch_count += 1
                step_tqdm = iter(self.tr_loader)
                if self.use_tqdm:
                    step_tqdm_legacy = step_tqdm = tqdm(
                        step_tqdm,
                        total=len(self.tr_loader),
                        position=1,
                        leave=False,
                    )
                for (x_batch, y_batch), index_batch in step_tqdm:
                    self._step_count += 1
                    with timing_context(self, "collate batch", enable=self.timing):
                        batch = self._collate_batch(x_batch, y_batch)
                    with amp_autocast_context(self._use_amp):
                        with timing_context(self, "model.forward", enable=self.timing):
                            forward_results = self.model(batch)
                        with timing_context(self, "loss.forward", enable=self.timing):
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
                    with timing_context(self, "loss.backward", enable=self.timing):
                        loss = loss_dict["loss"]
                        if self._use_amp:
                            loss = self.scaler.scale(loss)  # type: ignore
                        loss.backward()
                    if self._clip_norm > 0.0:
                        with timing_context(self, "clip_norm_step", enable=self.timing):
                            self._clip_norm_step()
                    with timing_context(self, "optimizer_step", enable=self.timing):
                        self._optimizer_step()
                    if self.ema_decay is not None:
                        with timing_context(self, "EMA", enable=self.timing):
                            self.ema_decay()
                    with timing_context(self, "monitor_step", enable=self.timing):
                        terminate = self._monitor_step()
                    if terminate:
                        break
            except KeyboardInterrupt:
                self.log_msg(  # type: ignore
                    "keyboard interrupted",
                    self.error_prefix,
                    msg_level=logging.ERROR,
                )
                terminate = True
            if terminate:
                if os.path.isdir(self.checkpoint_folder):
                    self.log_msg(  # type: ignore
                        "rolling back to the best checkpoint",
                        self.info_prefix,
                        3,
                    )
                    self.restore_checkpoint()
                break
            if self.use_tqdm:
                assert self._epoch_tqdm is not None
                self._epoch_tqdm.total = self.num_epoch
                self._epoch_tqdm.update()
        if self.use_tqdm:
            if step_tqdm_legacy is not None:
                step_tqdm_legacy.close()
            assert self._epoch_tqdm is not None
            self._epoch_tqdm.close()
        self._step_count = self._epoch_count = -1
        self._get_metrics()

    def predict(
        self,
        loader: DataLoader,
        return_all: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        predictions = self._predict(loader, **kwargs)
        if return_all:
            return predictions
        return predictions["predictions"]

    @staticmethod
    def _filter_checkpoints(folder: str) -> Dict[int, str]:
        checkpoints = {}
        for file in os.listdir(folder):
            if file.startswith("trainer_") and file.endswith(".pt"):
                step = int(os.path.splitext(file)[0].split("_")[1])
                checkpoints[step] = file
        return checkpoints

    def save_checkpoint(self, folder: str = None) -> None:
        if folder is None:
            folder = self.checkpoint_folder
        if self.max_snapshot_num > 0:
            checkpoints = self._filter_checkpoints(folder)
            if len(checkpoints) >= self.max_snapshot_num - 1:
                for key in sorted(checkpoints)[: -self.max_snapshot_num + 1]:
                    os.remove(os.path.join(folder, checkpoints[key]))
        file = f"trainer_{self._step_count}.pt"
        torch.save(self.state_dict(), os.path.join(folder, file))

    def restore_checkpoint(self, folder: str = None) -> "Trainer":
        if folder is None:
            folder = self.checkpoint_folder
        checkpoints = self._filter_checkpoints(folder)
        if not checkpoints:
            self.log_msg(  # type: ignore
                f"no trainer file found in {self.checkpoint_folder}",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            return self
        latest_checkpoint = checkpoints[sorted(checkpoints)[-1]]
        trainer_file = os.path.join(folder, latest_checkpoint)
        self.log_msg(  # type: ignore
            f"restoring from {trainer_file}",
            self.info_prefix,
            4,
        )
        self.load_state_dict(torch.load(trainer_file, map_location=self.model.device))
        return self


__all__ = [
    "to_prob",
    "amp_autocast_context",
    "Trainer",
]
