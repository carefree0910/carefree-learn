import os
import json
import math
import torch
import optuna
import inspect
import logging

import numpy as np

from typing import *
from tqdm import tqdm
from trains import Logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from cftool.ml import Metrics
from cftool.ml import Tracker
from cftool.ml import ScalarEMA
from cftool.misc import timestamp
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import fix_float_to_length
from cftool.misc import timing_context
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cfdata.tabular import DataLoader

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..misc.toolkit import *
from ..modules import optimizer_dict
from ..modules import scheduler_dict
from ..models.base import ModelBase
from ..pipeline.inference import Inference


trains_logger: Optional[Logger] = None


class IntermediateResults(NamedTuple):
    metrics: Dict[str, float]
    weighted_scores: Dict[str, float]

    @property
    def final_score(self) -> float:
        return sum(self.weighted_scores.values()) / len(self.weighted_scores)


class Trainer(LoggingMixin):
    pt_prefix = "model_"
    scores_file = "scores.json"

    def __init__(
        self,
        model: ModelBase,
        inference: Inference,
        trial: Optional[optuna.trial.Trial],
        tracker: Tracker,
        pipeline_config: Dict[str, Any],
        verbose_level: int,
        is_loading: bool,
    ):
        self.trial = trial
        self.tracker = tracker
        self.inference = inference
        self._init_config(pipeline_config, is_loading)
        self.model = model
        self.checkpoint_scores: Dict[str, float] = {}
        self.tr_loader_copy: Optional[DataLoader] = None
        self.final_results: Optional[IntermediateResults] = None
        self._verbose_level = verbose_level
        self._use_grad_in_predict = False
        self.onnx: Optional[Any] = None

    def _init_config(self, pipeline_config: Dict[str, Any], is_loading: bool) -> None:
        self._pipeline_config = pipeline_config
        self.timing = self._pipeline_config["use_timing_context"]
        self.use_tqdm = self._pipeline_config["use_tqdm"]
        self.config = pipeline_config.setdefault("trainer_config", {})
        self.update_bt_runtime = self.config.setdefault(
            "update_binary_threshold_at_runtime", False
        )
        self.min_epoch = int(self.config.setdefault("min_epoch", 0))
        num_epoch = self.config.setdefault("num_epoch", max(40, self.min_epoch))
        max_epoch = self.config.setdefault("max_epoch", max(200, num_epoch))
        self.num_epoch, self.max_epoch = map(int, [num_epoch, max_epoch])
        self.max_snapshot_file = int(self.config.setdefault("max_snapshot_file", 5))
        self.min_num_sample = self.config.setdefault("min_num_sample", 3000)
        self._snapshot_start_step = self.config.setdefault("snapshot_start_step", None)
        num_step_per_snapshot = self.config.setdefault("num_step_per_snapshot", 0)
        num_snapshot_per_epoch = self.config.setdefault("num_snapshot_per_epoch", 2)
        max_step_per_snapshot = self.config.setdefault("max_step_per_snapshot", 1000)
        plateau_start = self.config.setdefault("plateau_start_snapshot", num_epoch)
        self._num_step_per_snapshot = int(num_step_per_snapshot)
        self.num_snapshot_per_epoch = int(num_snapshot_per_epoch)
        self.max_step_per_snapshot = int(max_step_per_snapshot)
        self.plateau_start = int(plateau_start)

        self._clip_norm = self.config.setdefault("clip_norm", 0.0)
        self._use_amp = self.config.setdefault("use_amp", False)
        self.scaler = None if amp is None or not self._use_amp else amp.GradScaler()

        self._logging_path_ = pipeline_config["_logging_path_"]
        self.logging_folder = pipeline_config["logging_folder"]
        default_checkpoint_folder = os.path.join(self.logging_folder, "checkpoints")
        self.checkpoint_folder = self.config.setdefault(
            "checkpoint_folder", default_checkpoint_folder
        )
        Saving.prepare_folder(self, self.checkpoint_folder)

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
        opt = optimizer_base(parameters, **optimizer_config)
        self.optimizers[params_name] = opt
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
                if self._snapshot_start_step is not None:
                    self._snapshot_start_step += warmup_step
                else:
                    self.min_num_sample += self.tr_loader.batch_size * warmup_step
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
        if self.inference.use_binary_threshold:
            if "acc" in metric_types and self.inference.binary_metric is None:
                self.inference.binary_metric = "acc"
        self._metrics_need_loss = False
        self.metrics: Dict[str, Optional[Metrics]] = {}
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
    def snapshot_start_step(self) -> int:
        if self._snapshot_start_step is not None:
            return self._snapshot_start_step
        return int(math.ceil(self.min_num_sample / self.tr_loader.batch_size))

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

    @property
    def validation_loader(self) -> DataLoader:
        if self.cv_loader is None:
            return self.tr_loader_copy
        return self.cv_loader

    @property
    def validation_loader_name(self) -> str:
        loader = self.validation_loader
        return "tr" if loader is self.tr_loader_copy else "cv"

    @property
    def binary_threshold_loader(self) -> DataLoader:
        if self.cv_loader is not None and len(self.cv_loader.data) >= 1000:
            return self.cv_loader
        return self.tr_loader_copy

    @property
    def binary_threshold_loader_name(self) -> str:
        loader = self.binary_threshold_loader
        return "tr" if loader is self.tr_loader_copy else "cv"

    # core

    def _to_tqdm(self, loader: DataLoader) -> Union[tqdm, DataLoader]:
        if not self.use_tqdm:
            return loader
        return tqdm(loader, total=len(loader), leave=False, position=2)

    def _clip_norm_step(self) -> None:
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self._clip_norm
        )

    def _optimizer_step(self) -> None:
        self.model._optimizer_step(self.optimizers, self.scaler)

    # return whether we need to terminate
    def _monitor_step(self) -> bool:
        if self._step_count % self.num_step_per_snapshot == 0:

            with timing_context(self, "monitor.binary_threshold", enable=self.timing):
                rs = None
                if self.update_bt_runtime and self.start_snapshot:
                    inference = self.inference
                    if inference.need_binary_threshold:
                        loader = self.binary_threshold_loader
                        loader_name = self.binary_threshold_loader_name
                        rs = inference.generate_binary_threshold(loader, loader_name)

            with timing_context(self, "monitor.get_metrics", enable=self.timing):
                intermediate = self._get_metrics(rs)
                if self.start_monitor_plateau:
                    if not self._monitor.plateau_flag:
                        self.log_msg(  # type: ignore
                            "start monitoring plateau",
                            self.info_prefix,
                            2,
                        )
                    self._monitor.plateau_flag = True

            with timing_context(self, "monitor.core", enable=self.timing):
                if self.start_snapshot:
                    score = intermediate.final_score
                    if self.trial is not None:
                        self.trial.report(score, step=self._step_count)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()
                    if self._monitor.check_terminate(score):
                        return True
                    for key, scheduler in self.schedulers.items():
                        if scheduler is not None:
                            kwargs = {}
                            if key in self.schedulers_requires_metric:
                                kwargs["metrics"] = score
                            scheduler.step(**shallow_copy_dict(kwargs))  # type: ignore

        return False

    def _get_metrics(
        self,
        binary_threshold_outputs: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> IntermediateResults:
        if self.cv_loader is None and self.tr_loader._num_siamese > 1:
            raise ValueError("cv set should be provided when num_siamese > 1")
        loader = self.validation_loader
        loader_name = self.validation_loader_name
        # predictions
        if binary_threshold_outputs is not None:
            labels, probabilities = binary_threshold_outputs
            if not self.model.output_probabilities:
                logits = None
            else:
                logits = probabilities
            predictions = self.inference.predict_with(probabilities)
        else:
            keys = ["logits", "predictions", "labels"]
            results = self.inference.predict(
                loader=loader,
                loader_name=loader_name,
                return_all=True,
            )
            probabilities = None
            logits, predictions, labels = map(results.get, keys)
        # losses
        loss_values = None
        if self._metrics_need_loss:
            loader = self._to_tqdm(loader)
            forward_dicts, loss_dicts, labels = [], [], []
            for (x_batch, y_batch), batch_indices in loader:
                labels.append(y_batch)
                batch = self.inference.collate_batch(x_batch, y_batch)
                with eval_context(self.model):
                    forward_dicts.append(self.model(batch, batch_indices, loader_name))
                    loss_dicts.append(
                        self.model.loss_function(
                            batch,
                            batch_indices,
                            forward_dicts[-1],
                        )
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
                    if not metric_ins.requires_prob:
                        assert isinstance(predictions, np.ndarray)
                        metric_predictions = predictions
                    else:
                        if logits is None and probabilities is None:
                            msg = "`logits` should be returned in `inference.predict`"
                            raise ValueError(msg)
                        if self.model.output_probabilities:
                            metric_predictions = logits
                        else:
                            if logits is None:
                                metric_predictions = probabilities
                            else:
                                metric_predictions = to_prob(logits)
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
        weighted_scores = {
            k: v * signs[k] * self.metrics_weights[k]
            for k, v in metrics_for_scoring.items()
        }
        rs = IntermediateResults(metrics, weighted_scores)

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
            f"score : {fix_float_to_length(rs.final_score, 8)} |"
        )
        with open(self._log_file, "a") as f:
            f.write(f"{msg}\n")
        self.log_msg(msg, verbose_level=None)  # type: ignore

        return rs

    # api

    def fit(
        self,
        tr_loader: DataLoader,
        tr_loader_copy: DataLoader,
        cv_loader: DataLoader,
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
    ) -> None:
        self.tr_loader = tr_loader
        self.tr_loader_copy = tr_loader_copy
        self.cv_loader = cv_loader
        # sample weights
        if tr_weights is not None:
            tr_weights = to_torch(tr_weights)
        if cv_weights is not None:
            cv_weights = to_torch(cv_weights)
        self.tr_weights, self.cv_weights = tr_weights, cv_weights
        # optimizer
        self._init_optimizers()
        # metrics
        self._init_metrics()
        # monitor
        self._monitor = TrainMonitor(1).register_trainer(self)
        # train
        self.model.info()
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
                for (x_batch, y_batch), batch_indices in step_tqdm:
                    self._step_count += 1
                    with timing_context(self, "collate batch", enable=self.timing):
                        batch = self.inference.collate_batch(x_batch, y_batch)
                    with amp_autocast_context(self._use_amp):
                        with timing_context(self, "model.forward", enable=self.timing):
                            forward_results = self.model(batch, batch_indices, "tr")
                        with timing_context(self, "loss.forward", enable=self.timing):
                            loss_dict = self.model.loss_function(
                                batch,
                                batch_indices,
                                forward_results,
                            )
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
                    if self.model.use_ema:
                        with timing_context(self, "EMA", enable=self.timing):
                            self.model.apply_ema()
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
        rs = None
        if self.inference.need_binary_threshold:
            loader = self.binary_threshold_loader
            loader_name = self.binary_threshold_loader_name
            rs = self.inference.generate_binary_threshold(loader, loader_name)
        self.final_results = self._get_metrics(rs)

    def _sorted_checkpoints(self, folder: str, use_external_scores: bool) -> List[str]:
        # better checkpoints will be placed earlier
        # which means `checkpoints[0]` is the best checkpoint
        if not use_external_scores:
            scores = self.checkpoint_scores
        else:
            scores_path = os.path.join(folder, self.scores_file)
            if not os.path.isfile(scores_path):
                return []
            with open(scores_path, "r") as f:
                scores = json.load(f)
        files = [
            file
            for file in os.listdir(folder)
            if file.startswith(self.pt_prefix) and file.endswith(".pt")
        ]
        scores_list = [scores.get(file, -math.inf) for file in files]
        sorted_indices = np.argsort(scores_list)[::-1]
        return [files[i] for i in sorted_indices]

    def save_checkpoint(self, score: float, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = self.checkpoint_folder
        # leave top_k snapshots only
        if self.max_snapshot_file > 0:
            checkpoints = self._sorted_checkpoints(folder, False)
            if len(checkpoints) >= self.max_snapshot_file:
                for file in checkpoints[self.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        file = f"{self.pt_prefix}{self._step_count}.pt"
        torch.save(self.model.state_dict(), os.path.join(folder, file))
        # scores
        self.checkpoint_scores[file] = score
        with open(os.path.join(folder, self.scores_file), "w") as f:
            json.dump(self.checkpoint_scores, f)

    def restore_checkpoint(self, folder: str = None) -> "Trainer":
        if folder is None:
            folder = self.checkpoint_folder
        checkpoints = self._sorted_checkpoints(folder, True)
        if not checkpoints:
            self.log_msg(  # type: ignore
                f"no model file found in {self.checkpoint_folder}",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            return self
        best_checkpoint = checkpoints[0]
        model_file = os.path.join(folder, best_checkpoint)
        self.log_msg(  # type: ignore
            f"restoring from {model_file}",
            self.info_prefix,
            4,
        )
        states = torch.load(model_file, map_location=self.model.device)
        self.model.load_state_dict(states)
        return self


__all__ = ["Trainer", "IntermediateResults"]
