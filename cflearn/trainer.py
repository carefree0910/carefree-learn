import os
import json
import math
import torch
import optuna
import inspect
import logging

import numpy as np

from typing import *
from abc import abstractmethod
from abc import ABC
from tqdm.autonotebook import tqdm
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
from cftool.misc import Incrementer
from cftool.misc import LoggingMixin
from cfdata.tabular import DataLoader

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from .misc.toolkit import *
from .configs import Environment
from .modules import optimizer_dict
from .modules import scheduler_dict
from .inference import Inference
from .models.base import ModelBase


class IntermediateResults(NamedTuple):
    metrics: Dict[str, float]
    weighted_scores: Dict[str, float]
    use_decayed: bool
    decayed_metrics: Dict[str, float]

    @property
    def final_score(self) -> float:
        return sum(self.weighted_scores.values()) / len(self.weighted_scores)


class TrainerState:
    def __init__(self, trainer_config: Dict[str, Any]):
        # properties
        self.step = self.epoch = 0
        self.batch_size: int
        self.num_step_per_epoch: int
        # settings
        self.config = trainer_config
        self.min_epoch = self.config["min_epoch"]
        self.num_epoch = self.config["num_epoch"]
        self.max_epoch = self.config["max_epoch"]
        self.max_snapshot_file = int(self.config.setdefault("max_snapshot_file", 5))
        self.min_num_sample = self.config.setdefault("min_num_sample", 3000)
        self._snapshot_start_step = self.config.setdefault("snapshot_start_step", None)
        num_step_per_snapshot = self.config.setdefault("num_step_per_snapshot", 0)
        num_snapshot_per_epoch = self.config.setdefault("num_snapshot_per_epoch", 2)
        max_step_per_snapshot = self.config.setdefault("max_step_per_snapshot", 1000)
        plateau_start = self.config.setdefault("plateau_start_snapshot", self.num_epoch)
        self._num_step_per_snapshot = int(num_step_per_snapshot)
        self.num_snapshot_per_epoch = int(num_snapshot_per_epoch)
        self.max_step_per_snapshot = int(max_step_per_snapshot)
        self.plateau_start = int(plateau_start)

    def inject_loader(self, loader: DataLoader) -> None:
        self.batch_size = loader.batch_size
        self.num_step_per_epoch = len(loader)

    @property
    def snapshot_start_step(self) -> int:
        if self._snapshot_start_step is not None:
            return self._snapshot_start_step
        return int(math.ceil(self.min_num_sample / self.batch_size))

    @property
    def num_step_per_snapshot(self) -> int:
        if self._num_step_per_snapshot > 0:
            return self._num_step_per_snapshot
        return max(
            1,
            min(
                self.max_step_per_snapshot,
                int(self.num_step_per_epoch / self.num_snapshot_per_epoch),
            ),
        )

    @property
    def should_train(self) -> bool:
        return self.epoch < self.num_epoch

    @property
    def should_monitor(self) -> bool:
        return self.step % self.num_step_per_snapshot == 0

    @property
    def should_log_metrics_msg(self) -> bool:
        min_period = self.max_step_per_snapshot / 3
        min_period = math.ceil(min_period / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step and self.epoch > self.min_epoch

    @property
    def should_start_monitor_plateau(self) -> bool:
        return self.step >= self.plateau_start * self.num_step_per_snapshot

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch and self.epoch < self.max_epoch

    @property
    def reached_max_epoch(self) -> bool:
        return self.epoch == self.max_epoch


# Should define `TrainerState` as `self.state`
class MonitoredMixin(ABC, LoggingMixin):
    @abstractmethod
    def on_save_checkpoint(self, score: float) -> None:
        pass


class TrainMonitor:
    """
    Util class to monitor training process of a neural network
    * If overfitting, it will tell the model to early-stop
    * If underfitting, it will tell the model to extend training process
    * If better performance acquired, it will tell the model to save a checkpoint
    * If performance sticks on a plateau, it will tell the model to stop training (to save time)

    Warnings
    ----------
    * Performance should represent 'score', i.e. the higher the better
    * Performance MUST be evaluated on the cross validation dataset instead of the training set if possible
    * `register_trainer` method MUST be called before monitoring
    * instance passed to`register_trainer` method MUST be a subclass of `LoggingMixin`,
      and must include `_epoch_count`, `num_epoch`, `max_epoch` attributes and
      `save_checkpoint` method

    Parameters
    ----------
    monitored : MonitoredMixin, monitored instance
    num_scores_per_snapshot : int, indicates snapshot frequency
        * `TrainMonitor` will perform a snapshot every `num_scores_per_snapshot` scores are recorded
    history_ratio : float, indicates the ratio of the history's window width
        * history window width will be `num_scores_per_snapshot` * `history_ratio`
    tolerance_ratio : float, indicates the ratio of tolerance
        * tolerance base will be `num_scores_per_snapshot` * `tolerance_ratio`
        * judgements of 'overfitting' and 'performance sticks on a plateau' will based on 'tolerance base'
    extension : int, indicates how much epoch to extend when underfitting occurs
    std_floor : float, indicates the floor of history's std used for judgements
    std_ceiling : float, indicates the ceiling of history's std used for judgements
    aggressive : bool, indicates the strategy of monitoring
        * True  : it will tell the model to save every checkpoint when better metric is reached
        * False : it will be more careful since better metric may lead to
        more seriously over-fitting on cross validation set

    Examples
    ----------
    >>> from cftool.ml import Metrics
    >>>
    >>> x, y, model = ...
    >>> metric = Metrics("mae")
    >>> monitor = TrainMonitor.monitor(model)
    >>> n_epoch, epoch_count = 20, 0
    >>> while epoch_count <= n_epoch:
    >>>     model.train()
    >>>     predictions = model.predict(x)
    >>>     score = metric.metric(y, predictions) * metric.sign
    >>>     if monitor.check_terminate(score):
    >>>         break

    """

    def __init__(
        self,
        monitored: MonitoredMixin,
        num_scores_per_snapshot: int = 1,
        history_ratio: int = 3,
        tolerance_ratio: int = 2,
        extension: int = 5,
        std_floor: float = 0.001,
        std_ceiling: float = 0.01,
        aggressive: bool = False,
    ):
        self.monitored = monitored
        self.num_scores_per_snapshot = num_scores_per_snapshot
        self.num_history = int(num_scores_per_snapshot * history_ratio)
        self.num_tolerance = int(num_scores_per_snapshot * tolerance_ratio)
        self.extension = extension
        self.is_aggressive = aggressive
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores: List[float] = []
        self.plateau_flag = False
        self._is_best: Optional[bool] = None
        self._running_best: Optional[float] = None
        self._descend_increment = self.num_history * extension / 30.0
        self._incrementer = Incrementer(self.num_history)

        self._over_fit_performance = math.inf
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._plateau_counter = self.over_fitting_flag = 0.0
        self.info: Dict[str, Any] = {
            "terminate": False,
            "save_checkpoint": False,
            "save_best": aggressive,
            "info": None,
        }

    @property
    def state(self) -> TrainerState:
        return self.monitored.state

    @property
    def log_msg(self) -> Callable:
        return self.monitored.log_msg

    @property
    def plateau_threshold(self) -> int:
        return 6 * self.num_tolerance * self.num_history

    def _update_running_info(self, last_score: float) -> float:
        self._incrementer.update(last_score)
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0.0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > last_score:
            improvement = 0
            self._is_best = False
        else:
            improvement = last_score - self._running_best
            self._running_best = last_score
            self._is_best = True
        return improvement

    def _handle_overfitting(self, last_score: float, res: float, std: float) -> None:
        if self._descend_counter == 0.0:
            self.info["save_best"] = True
            self._over_fit_performance = last_score
        self._descend_counter += min(self.num_tolerance / 3, -res / std)
        self.log_msg(
            f"descend counter updated : {self._descend_counter:6.4f}",
            prefix=self.monitored.info_prefix,
            verbose_level=6,
            msg_level=logging.DEBUG,
        )
        self.over_fitting_flag = 1

    def _handle_recovering(
        self,
        improvement: float,
        last_score: float,
        res: float,
        std: float,
    ) -> None:
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
                assert self._running_best is not None
                if last_score > self._running_best - std:
                    self._plateau_counter //= 2
                    self.info["save_checkpoint"] = True
                    self.info["info"] = (
                        f"current snapshot ({len(self._scores)}) seems to be working well, "
                        "saving checkpoint in case we need to restore"
                    )
            self.over_fitting_flag = 0
        if self._descend_counter > 0:
            self._descend_counter = max(new_counter, 0)
            self.log_msg(
                f"descend counter updated : {self._descend_counter:6.4f}",
                prefix=self.monitored.info_prefix,
                verbose_level=6,
                msg_level=logging.DEBUG,
            )

    def _handle_is_best(self) -> None:
        if self._is_best:
            self.info["terminate"] = False
            if self.info["save_best"]:
                self._plateau_counter //= 2
                self.info["save_checkpoint"] = True
                self.info["save_best"] = self.is_aggressive
                self.info["info"] = (
                    f"current snapshot ({len(self._scores)}) leads to best result we've ever had, "
                    "saving checkpoint since "
                )
                if self.over_fitting_flag:
                    self.info["info"] += "we've suffered from over-fitting"
                else:
                    self.info["info"] += "performance has improved significantly"

    def _handle_period(self, last_score: float) -> None:
        if self.is_aggressive:
            return
        if (
            len(self._scores) % self.num_scores_per_snapshot == 0
            and last_score > self._best_checkpoint_performance
        ):
            self._best_checkpoint_performance = last_score
            self._plateau_counter //= 2
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = (
                f"current snapshot ({len(self._scores)}) leads to best checkpoint we've ever had, "
                "saving checkpoint in case we need to restore"
            )

    def _punish_extension(self) -> None:
        self.plateau_flag = True
        self._descend_counter += self._descend_increment

    def _handle_trainer_terminate(self, score: float) -> bool:
        if self.info["terminate"]:
            self.log_msg(
                f"early stopped at n_epoch={self.state.epoch} "
                f"due to '{self.info['info']}'",
                prefix=self.monitored.info_prefix,
            )
            return True
        if self.info["save_checkpoint"]:
            self.log_msg(f"{self.info['info']}", self.monitored.info_prefix, 3)
            self.monitored.on_save_checkpoint(score)
        if self.state.should_extend_epoch and not self.info["terminate"]:
            self._punish_extension()
            new_epoch = self.state.num_epoch + self.extension
            self.state.num_epoch = min(new_epoch, self.state.max_epoch)
            self.log_msg(
                f"extending num_epoch to {self.state.num_epoch}",
                prefix=self.monitored.info_prefix,
                verbose_level=3,
            )
        if self.state.reached_max_epoch:
            if not self.info["terminate"]:
                self.log_msg(
                    "model seems to be under-fitting but max_epoch reached, "
                    "increasing max_epoch may improve performance.",
                    self.monitored.info_prefix,
                )
            return True
        return False

    def check_terminate(self, new_score: float) -> bool:
        self._scores.append(new_score)
        n_history = min(self.num_history, len(self._scores))
        if math.isnan(new_score):
            self.info["terminate"] = True
            self.info["info"] = "nan metric encountered"
        elif n_history != 1:
            improvement = self._update_running_info(new_score)
            self.info["save_checkpoint"] = False
            mean, std = self._incrementer.mean, self._incrementer.std
            std = min(std, self.std_ceiling)
            plateau_updated = False
            if std < self.std_floor:
                if self.plateau_flag:
                    increment = self.std_floor / max(std, self.std_floor / 6)
                    self._plateau_counter += increment
                    plateau_updated = True
            else:
                if self._plateau_counter > 0:
                    self._plateau_counter = max(self._plateau_counter - 1, 0)
                    plateau_updated = True
                res = new_score - mean
                if res < -std and new_score < self._over_fit_performance - std:
                    self._handle_overfitting(new_score, res, std)
                elif res > std:
                    self._handle_recovering(improvement, new_score, res, std)
            if plateau_updated:
                self.log_msg(
                    f"plateau counter updated : {self._plateau_counter:>6.4f} "
                    f"/ {self.plateau_threshold}",
                    prefix=self.monitored.info_prefix,
                    verbose_level=6,
                    msg_level=logging.DEBUG,
                )
            if self._plateau_counter >= self.plateau_threshold:
                self.info["info"] = "performance not improving"
                self.info["terminate"] = True
            else:
                if self._descend_counter >= self.num_tolerance:
                    self.info["info"] = "over-fitting"
                    self.info["terminate"] = True
                else:
                    self._handle_is_best()
                    self._handle_period(new_score)
                    if self.info["save_checkpoint"]:
                        self.info["info"] += " (plateau counter cleared)"
                        self._plateau_counter = 0
        return self._handle_trainer_terminate(new_score)

    @classmethod
    def monitor(cls, monitored: MonitoredMixin, **kwargs: Any) -> "TrainMonitor":
        return cls(monitored, **kwargs)


class Trainer(MonitoredMixin):
    pt_prefix = "model_"
    scores_file = "scores.json"

    def __init__(
        self,
        model: ModelBase,
        inference: Inference,
        environment: Environment,
        is_loading: bool,
    ):
        # common
        self.model = model
        self.inference = inference
        self.environment = environment
        self.trial = environment.trial
        self.device = environment.device
        if environment.tracker_config is None:
            self.tracker = None
        else:
            self.tracker = Tracker(**environment.tracker_config)
        self.checkpoint_scores: Dict[str, float] = {}
        self.tr_loader_copy: Optional[DataLoader] = None
        self.final_results: Optional[IntermediateResults] = None
        self._use_grad_in_predict = False
        self.onnx: Optional[Any] = None
        # config based
        self.timing = environment.use_timing_context
        self.config = environment.trainer_config
        self._verbose_level = environment.verbose_level
        self.update_bt_runtime = self.update_binary_threshold_at_runtime
        self.scaler = None if amp is None or not self.use_amp else amp.GradScaler()
        self.state = TrainerState(self.config)
        Saving.prepare_folder(self, self.checkpoint_folder)

    def __getattr__(self, item: str) -> Any:
        value = self.config.get(item)
        if value is not None:
            return value
        return self.environment.config[item]

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
            optimizer = opt_setting["optimizer"]
            optimizer_config = opt_setting["optimizer_config"]
            scheduler = opt_setting["scheduler"]
            scheduler_config = opt_setting["scheduler_config"]
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
                        * self.state.plateau_start
                        * self.state.num_step_per_snapshot
                    ),
                )
                warmup_step = scheduler_config.setdefault(
                    "warmup_step", default_warm_up_step
                )
                self.state.plateau_start += int(
                    warmup_step / self.state.num_step_per_snapshot
                )
                if self.state._snapshot_start_step is not None:
                    self.state._snapshot_start_step += warmup_step
                else:
                    self.state.min_num_sample += self.tr_loader.batch_size * warmup_step
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
                max(
                    10,
                    self.state.snapshot_start_step // self.state.num_step_per_snapshot,
                ),
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
            if scheduler is None:
                continue
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

    def _clip_norm_step(self) -> None:
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_norm
        )

    def _optimizer_step(self) -> None:
        self.model._optimizer_step(self.optimizers, self.scaler)

    @staticmethod
    def _metric_verbose(k: str, intermediate: IntermediateResults) -> str:
        metric_str = fix_float_to_length(intermediate.metrics[k], 8)
        if not intermediate.use_decayed:
            return metric_str
        decayed = intermediate.decayed_metrics[k]
        return f"{metric_str} (ema: {fix_float_to_length(decayed, 8)})"

    def _log_metrics_msg(self, intermediate: IntermediateResults) -> None:
        core = " | ".join(
            [
                f"{k} : {self._metric_verbose(k, intermediate)}"
                for k in sorted(intermediate.metrics)
            ]
        )
        msg = (
            f"| epoch {self.state.epoch:^4d} - "
            f"step {self.state.step:^6d} | {core} | "
            f"score : {fix_float_to_length(intermediate.final_score, 8)} |"
        )
        with open(self._log_file, "a") as f:
            f.write(f"{msg}\n")
        self.log_msg(msg, verbose_level=None)  # type: ignore

    # return whether we need to terminate
    def _monitor_step(self) -> bool:
        if self.state.should_monitor:

            with timing_context(self, "monitor.binary_threshold", enable=self.timing):
                rs = None
                if self.update_bt_runtime and self.state.should_start_snapshot:
                    inference = self.inference
                    if inference.need_binary_threshold:
                        loader = self.binary_threshold_loader
                        loader_name = self.binary_threshold_loader_name
                        rs = inference.generate_binary_threshold(loader, loader_name)

            with timing_context(self, "monitor.get_metrics", enable=self.timing):
                intermediate = self._get_metrics(rs)
                if self.state.should_start_monitor_plateau:
                    if not self._monitor.plateau_flag:
                        self.log_msg(  # type: ignore
                            "start monitoring plateau",
                            self.info_prefix,
                            3,
                        )
                    self._monitor.plateau_flag = True

            with timing_context(self, "monitor.logging", enable=self.timing):
                if self.state.should_log_metrics_msg:
                    self._log_metrics_msg(intermediate)

            if self.state.should_start_snapshot:
                timing_name = "monitor.prune_trial"
                with timing_context(self, timing_name, enable=self.timing):
                    score = intermediate.final_score
                    if self.trial is not None:
                        self.trial.report(score, step=self.state.step)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()
                timing_name = "monitor.check_terminate"
                with timing_context(self, timing_name, enable=self.timing):
                    if self._monitor.check_terminate(score):
                        return True
                timing_name = "monitor.scheduler"
                with timing_context(self, timing_name, enable=self.timing):
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
            loader = self.inference.to_tqdm(loader)
            loss_dicts = []
            for (x_batch, y_batch), batch_indices in loader:
                batch = self.inference.collate_batch(x_batch, y_batch)
                with eval_context(self.model):
                    loss_dicts.append(
                        self.model.loss_function(
                            batch,
                            batch_indices,
                            self.model(
                                batch,
                                batch_indices,
                                loader_name,
                                self.state.step,
                            ),
                            self.state.step,
                        )
                    )
            losses = collate_tensor_dicts(loss_dicts)
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
            if self.metrics_decay is not None and self.state.should_start_snapshot:
                use_decayed = True
                decayed = self.metrics_decay[metric_type].update("metric", sub_metric)
                decayed_metrics[metric_type] = decayed
        metrics_for_scoring = decayed_metrics if use_decayed else metrics
        if self._epoch_tqdm is not None:
            self._epoch_tqdm.set_postfix(metrics_for_scoring)
        if self.tracker is not None:
            for name, value in metrics_for_scoring.items():
                if self.tracker is not None:
                    self.tracker.track_scalar(name, value, iteration=self.state.step)
        weighted_scores = {
            k: v * signs[k] * self.metrics_weights[k]
            for k, v in metrics_for_scoring.items()
        }
        return IntermediateResults(
            metrics,
            weighted_scores,
            use_decayed,
            decayed_metrics,
        )

    def on_save_checkpoint(self, score: float) -> None:
        self.save_checkpoint(score)

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
        self.state.inject_loader(tr_loader)
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
        self._monitor = TrainMonitor.monitor(self)
        # train
        self.model.info()
        terminate = False
        tuple(
            map(
                lambda n: os.makedirs(n, exist_ok=True),
                [self.logging_folder, self.checkpoint_folder],
            )
        )
        log_name = f"{timestamp()}.txt"
        self._log_file: str = os.path.join(self.logging_folder, log_name)
        with open(self._log_file, "w"):
            pass
        step_tqdm_legacy = None
        self._epoch_tqdm: Optional[tqdm] = None
        if self.use_tqdm:
            self._epoch_tqdm = tqdm(list(range(self.state.num_epoch)), position=0)
        while self.state.should_train:
            try:
                self.state.epoch += 1
                step_tqdm = iter(self.tr_loader)
                if self.use_tqdm:
                    step_tqdm_legacy = step_tqdm = tqdm(
                        step_tqdm,
                        total=len(self.tr_loader),
                        position=1,
                        leave=False,
                    )
                for (x_batch, y_batch), batch_indices in step_tqdm:
                    self.state.step += 1
                    with timing_context(self, "collate batch", enable=self.timing):
                        batch = self.inference.collate_batch(x_batch, y_batch)
                    with amp_autocast_context(self.use_amp):
                        with timing_context(self, "model.forward", enable=self.timing):
                            forward_results = self.model(
                                batch,
                                batch_indices,
                                "tr",
                                self.state.step,
                            )
                        with timing_context(self, "loss.forward", enable=self.timing):
                            loss_dict = self.model.loss_function(
                                batch,
                                batch_indices,
                                forward_results,
                                self.state.step,
                            )
                    if self.tracker is not None:
                        for name, tensor in loss_dict.items():
                            value = tensor.item()
                            if self.tracker is not None:
                                self.tracker.track_scalar(
                                    f"tr_{name}",
                                    value,
                                    iteration=self.state.step,
                                )
                    with timing_context(self, "loss.backward", enable=self.timing):
                        loss = loss_dict["loss"]
                        if self.use_amp:
                            loss = self.scaler.scale(loss)  # type: ignore
                        loss.backward()
                    if self.clip_norm > 0.0:
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
                self._epoch_tqdm.total = self.state.num_epoch
                self._epoch_tqdm.update()
        if self.use_tqdm:
            if step_tqdm_legacy is not None:
                step_tqdm_legacy.close()
            assert self._epoch_tqdm is not None
            self._epoch_tqdm.close()
        rs = None
        if self.inference.need_binary_threshold:
            loader = self.binary_threshold_loader
            loader_name = self.binary_threshold_loader_name
            rs = self.inference.generate_binary_threshold(loader, loader_name)
        self.state.epoch = self.state.step = -1
        self.final_results = self._get_metrics(rs)
        self._log_metrics_msg(self.final_results)

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
        if self.state.max_snapshot_file > 0:
            checkpoints = self._sorted_checkpoints(folder, False)
            if len(checkpoints) >= self.state.max_snapshot_file:
                for file in checkpoints[self.state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        file = f"{self.pt_prefix}{self.state.epoch}.pt"
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
        states = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(states)
        return self


__all__ = [
    "IntermediateResults",
    "TrainerState",
    "MonitoredMixin",
    "TrainMonitor",
    "Trainer",
]
