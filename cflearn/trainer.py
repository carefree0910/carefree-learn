import os
import json
import math
import torch
import mlflow
import optuna
import getpass
import logging

import numpy as np
import torch.distributed as dist

from typing import *
from abc import abstractmethod
from abc import ABC
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from cftool.ml import Metrics
from cftool.ml import ScalarEMA
from cftool.misc import timestamp
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import fix_float_to_length
from cftool.misc import lock_manager
from cftool.misc import timing_context
from cftool.misc import Incrementer
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.tracking.fluent import _RUN_ID_ENV_VAR

from .misc.toolkit import *
from .types import tensor_dict_type
from .configs import Environment
from .modules import optimizer_dict
from .modules import scheduler_dict
from .protocol import StepOutputs
from .protocol import TrainerState
from .protocol import ModelProtocol
from .protocol import PrefetchLoader
from .protocol import InferenceOutputs
from .protocol import InferenceProtocol
from .protocol import DataLoaderProtocol
from .modules.schedulers import WarmupScheduler


class IntermediateResults(NamedTuple):
    metrics: Dict[str, float]
    weighted_metrics: Dict[str, float]
    weighted_scores: Dict[str, float]
    use_decayed: bool
    decayed_metrics: Dict[str, float]

    @property
    def final_score(self) -> float:
        return sum(self.weighted_scores.values()) / len(self.weighted_scores)


# Should define `TrainerState` as `self.state`
class MonitoredMixin(ABC, LoggingMixinWithRank):
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

    Parameters
    ----------
    monitored : MonitoredMixin, monitored instance
    patience : int, basically indicates the 'patience' of `TrainMonitor`
    history_ratio : float, indicates the ratio of the history's window width
        * history window width will be `patience` * `history_ratio`
    tolerance_ratio : float, indicates the ratio of tolerance
        * tolerance base will be `patience` * `tolerance_ratio`
        * judgements of 'overfitting' and 'performance sticks on a plateau' will based on 'tolerance base'
    extension : int, indicates how much epoch to extend when underfitting occurs
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
        *,
        patience: int = 4,
        history_ratio: float = 3.0,
        tolerance_ratio: float = 2.0,
        extension: int = 5,
        lazy: bool = False,
        aggressive: bool = False,
    ):
        self.monitored = monitored
        self.tolerance_ratio = tolerance_ratio
        self.num_history = int(round(patience * history_ratio))
        self.num_tolerance = int(round(patience * tolerance_ratio))
        self.plateau_threshold = int(round(patience * history_ratio * tolerance_ratio))
        self.extension = extension
        self.is_lazy = lazy
        self.is_aggressive = aggressive
        self._scores: List[float] = []
        self.plateau_flag = False
        self._is_best: Optional[bool] = None
        self._running_best: Optional[float] = None
        self._descend_increment = self.num_history * extension / 30.0
        self._incrementer = Incrementer(self.num_history)

        self._score_before_overfit = math.inf
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

    def _update_running_info(self, new_score: float) -> float:
        self._incrementer.update(new_score)
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0.0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > new_score:
            improvement = 0
            self._is_best = False
        else:
            improvement = new_score - self._running_best
            self._running_best = new_score
            self._is_best = True
        return improvement

    def _log_descend_counter(self, new_score: float, res: float, std: float) -> None:
        self.log_msg(
            f"descend counter updated : {self._descend_counter:6.4f}, "
            f"last_score: {new_score:8.6f}, res: {res:8.6f}, std: {std:8.6f}",
            prefix=self.monitored.info_prefix,
            verbose_level=6,
            msg_level=logging.DEBUG,
        )

    def _handle_overfitting(
        self,
        new_score: float,
        res: float,
        mean: float,
        std: float,
    ) -> None:
        if self._descend_counter == 0.0:
            self.info["save_best"] = True
            self._score_before_overfit = mean
        self._descend_counter += min(self.tolerance_ratio, max(0.0, -res / std - 1.0))
        self._log_descend_counter(new_score, res, std)
        self.over_fitting_flag = 1

    def _handle_recovering(
        self,
        improvement: float,
        new_score: float,
        res: float,
        std: float,
    ) -> None:
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std
        if self._descend_counter > 0 >= new_counter:
            self._score_before_overfit = math.inf
            if new_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = new_score
                assert self._running_best is not None
                if new_score > self._running_best - std:
                    self._plateau_counter //= 2
                    self.info["save_checkpoint"] = True
                    self.info["info"] = (
                        f"current snapshot ({len(self._scores)}) seems to be working well, "
                        "saving checkpoint in case we need to restore"
                    )
            self.over_fitting_flag = 0
        if self._descend_counter > 0:
            self._descend_counter = max(new_counter, 0)
            self._log_descend_counter(new_score, res, std)

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

    def _handle_period(self, new_score: float) -> None:
        if self.is_aggressive:
            return
        if new_score > self._best_checkpoint_performance:
            self._best_checkpoint_performance = new_score
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

    def _handle_trainer_terminate(self, new_score: float) -> bool:
        if self.info["terminate"] and not self.is_lazy:
            if not self.state.reached_min_epoch:
                self.log_msg(
                    f"'{self.info['info']}' was detected but min_epoch was not reached,"
                    " the training process will be continued",
                    prefix=self.monitored.warning_prefix,
                    verbose_level=3,
                )
                return False
            self.log_msg(
                f"early stopped at n_epoch={self.state.epoch} "
                f"due to '{self.info['info']}'",
                prefix=self.monitored.info_prefix,
                verbose_level=3,
            )
            return True
        if self.info["save_checkpoint"]:
            if self.monitored.is_rank_0:
                self.log_msg(f"{self.info['info']}", self.monitored.info_prefix, 3)
                self.monitored.on_save_checkpoint(new_score)
        if self.state.should_extend_epoch:
            if self.is_lazy and self.state.reached_min_epoch:
                return True
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
            log10 = math.log10(abs(mean))
            std_floor = 10.0 ** (log10 - 3.0)
            std_ceiling = 10.0 ** (log10 - 1.0)
            std = min(std, std_ceiling)
            plateau_updated = False
            if std < std_floor:
                if self.plateau_flag:
                    plateau_increment = std_floor / max(std, std_floor / 6.0)
                    self._plateau_counter += plateau_increment
                    plateau_updated = True
            else:
                if self._plateau_counter > 0:
                    self._plateau_counter = max(self._plateau_counter - 1, 0)
                    plateau_updated = True
                if math.isinf(self._score_before_overfit):
                    res = new_score - mean
                else:
                    res = new_score - self._score_before_overfit
                if res < -std:
                    self._handle_overfitting(new_score, res, mean, std)
                elif res > std:
                    self._handle_recovering(improvement, new_score, res, std)
            if plateau_updated:
                self.log_msg(
                    f"plateau counter updated : {self._plateau_counter:>6.4f} "
                    f"/ {self.plateau_threshold}, "
                    f"std: {std:8.6f}, std_floor: {std_floor:8.6f}",
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


class MonitorResults(NamedTuple):
    terminate: bool
    outputs: Optional[InferenceOutputs]


class TrainerCallback:
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def after_step(self, step_outputs: StepOutputs) -> None:
        pass

    def after_monitor(self, monitor_results: MonitorResults) -> None:
        pass


class TqdmSettings(NamedTuple):
    use_tqdm: bool
    use_step_tqdm: bool
    use_tqdm_in_cv: bool
    in_distributed: bool
    position: int
    desc: str


def _setup_ddp(
    rank: int,
    world_size: int,
    *,
    backend: str = "gloo",
    port: str = "12355",
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)


class Trainer(MonitoredMixin):
    callback_base = TrainerCallback

    def __init__(
        self,
        model: ModelProtocol,
        inference: InferenceProtocol,
        environment: Environment,
        is_loading: bool,
    ):
        # common
        self.model = model
        self.callback = self.callback_base(self)
        self.inference = inference
        self.environment = environment
        self.is_loading = is_loading
        self.trial = environment.trial
        self.device = environment.device
        self.is_rank_0 = environment.is_rank_0
        self._init_mlflow(environment)
        self.checkpoint_scores: Dict[str, float] = {}
        self.tr_loader_copy: Optional[PrefetchLoader] = None
        self.intermediate: Optional[IntermediateResults] = None
        self.intermediate_updated = False
        self.final_results: Optional[IntermediateResults] = None
        self._use_grad_in_predict = False
        self.onnx: Optional[Any] = None
        # config based
        self.timing = environment.use_timing_context
        self.config = environment.trainer_config
        self.tqdm_settings = TqdmSettings(
            self.use_tqdm,
            environment.pipeline_config.setdefault("use_step_tqdm", self.use_tqdm),
            environment.pipeline_config.setdefault("use_tqdm_in_cv", False),
            environment.pipeline_config.setdefault("in_distributed", False),
            environment.pipeline_config.setdefault("tqdm_position", 0),
            environment.pipeline_config.setdefault("tqdm_desc", "epoch"),
        )
        self._verbose_level = environment.verbose_level
        self.update_bt_runtime = self.update_binary_threshold_at_runtime
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.state = TrainerState(self.config)

    def __getattr__(self, item: str) -> Any:
        value = self.config.get(item)
        if value is not None:
            return value
        return self.environment.config[item]

    # mlflow

    def _init_mlflow(self, environment: Environment) -> None:
        self.run_id: Optional[str] = None
        self.mlflow_client: Optional[mlflow.tracking.MlflowClient] = None
        mlflow_config = environment.mlflow_config
        if mlflow_config is None or self.is_loading or not self.is_rank_0:
            return None

        model = self.model.__identifier__
        task_type = self.model.data.task_type.value
        task_name = mlflow_config.setdefault("task_name", f"{model}({task_type})")
        tracking_folder = mlflow_config.setdefault("tracking_folder", os.getcwd())
        tracking_folder = os.path.abspath(tracking_folder)
        tracking_dir = os.path.join(tracking_folder, "mlruns")
        with lock_manager(tracking_folder, ["mlruns"]):
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_uri = parse_uri(tracking_dir)
            self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
            experiment = self.mlflow_client.get_experiment_by_name(task_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = self.mlflow_client.create_experiment(task_name)

        run = None
        self.from_external = False
        if _RUN_ID_ENV_VAR in os.environ:
            existing_run_id = os.environ[_RUN_ID_ENV_VAR]
            del os.environ[_RUN_ID_ENV_VAR]
            try:
                run = self.mlflow_client.get_run(existing_run_id)
                self.from_external = True
            except MlflowException:
                print(
                    f"{self.warning_prefix}`run_id` is found in environment but "
                    "corresponding mlflow run does not exist. This might cause by "
                    "external calls."
                )
        if run is None:
            run_tags: Dict[str, Any] = mlflow_config.setdefault("run_tags", {})
            run_tags.setdefault(MLFLOW_USER, getpass.getuser())
            run_name = mlflow_config.setdefault("run_name", None)
            run_name_prefix = mlflow_config.setdefault("run_name_prefix", None)
            if run_name is not None:
                if run_name_prefix is not None:
                    run_name = f"{run_name_prefix}_{run_name}"
                run_tags.setdefault(MLFLOW_RUN_NAME, run_name)
            run = self.mlflow_client.create_run(experiment_id, tags=run_tags)
        self.run_id = run.info.run_id

        self.mlflow_params = mlflow_config.get("mlflow_params")

    def _prepare_log(self) -> None:
        tuple(
            map(
                lambda folder: os.makedirs(folder, exist_ok=True),
                [self.logging_folder, self.checkpoint_folder],
            )
        )
        log_name = f"{timestamp()}.txt"
        self._log_file: str = os.path.join(self.logging_folder, log_name)
        with open(self._log_file, "w"):
            pass
        if self.mlflow_client is None:
            return None
        mlflow_params = self.mlflow_params or self.environment.user_defined_config
        if not self.from_external:
            for key, value in mlflow_params.items():
                self.mlflow_client.log_param(self.run_id, key, value)

    def _log_scalars(self, metrics: Dict[str, float]) -> None:
        if self.mlflow_client is None:
            return None
        for key, value in metrics.items():
            self.mlflow_client.log_metric(self.run_id, key, value, step=self.state.step)

    def _log_artifacts(self) -> None:
        if self.mlflow_client is None:
            return None
        self.mlflow_client.log_artifacts(self.run_id, self.logging_folder)

    # ddp

    def _init_ddp(self) -> None:
        self.ddp_model = None
        if not self.ddp:
            return None
        # monitor
        monitor_config = self.config.setdefault("monitor_config", {})
        monitor_config["lazy"] = True
        # ddp setup
        rank = self.environment.rank
        world_size = self.environment.world_size
        assert isinstance(rank, int)
        if world_size is None:
            raise ValueError("`world_size` should be provided when `ddp` is used")
        ddp_config = self.config.setdefault("ddp_config", {})
        _setup_ddp(rank, world_size, **ddp_config)
        self.ddp_model = DDP(self.model.to(rank), device_ids=[rank])

    # init

    def default_lr_configs(
        self,
        optimizer: Optimizer,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        opt_lr = optimizer_config["lr"]
        # step
        step_default_cfg = {"step_size": 10 * self.state.num_step_per_epoch}
        # exponential
        exp_gamma = (0.1 ** 0.1) ** (1.0 / self.state.num_step_per_epoch)
        exp_default_cfg = {"gamma": exp_gamma}
        # cyclic
        cyclic_default_cfg = {
            "base_lr": opt_lr,
            "max_lr": 1.0e-8,
            "step_size_up": 10 * self.state.num_step_per_epoch,
            "gamma": exp_gamma,
        }
        if "momentum" not in optimizer.defaults:
            cyclic_default_cfg["cycle_momentum"] = False
        # cosine
        cosine_default_cfg = {
            "eta_min": 1.0e-8,
            "T_max": 10 * self.state.num_step_per_epoch,
        }
        # cosine restarts
        cosine_restarts_default_cfg = {
            "eta_min": 1.0e-8,
            "T_0": 10 * self.state.num_step_per_epoch,
        }
        # plateau
        plateau_default_cfg = {
            "mode": "max",
            "min_lr": 1.0e-8,
            "verbose": self._verbose_level >= 3,
            "patience": max(
                10 * self.state.num_step_per_snapshot,
                self.state.snapshot_start_step,
            ),
        }
        return {
            "step": step_default_cfg,
            "exponential": exp_default_cfg,
            "cyclic": cyclic_default_cfg,
            "cosine": cosine_default_cfg,
            "cosine_restarts": cosine_restarts_default_cfg,
            "plateau": plateau_default_cfg,
        }

    def _define_optimizer(
        self,
        params_name: str,
        optimizer_base: Type[Optimizer],
        optimizer_config: Dict[str, Any],
    ) -> Optimizer:
        if params_name == "all":
            parameters = self.model_for_training.parameters()
        else:
            attr = getattr(self.model_for_training, params_name)
            if not isinstance(attr, torch.nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        opt = optimizer_base(parameters, **optimizer_config)
        self.optimizers[params_name] = opt
        return opt

    def _init_optimizers(self) -> None:
        optimizers_settings = self.config.setdefault("optimizers", {"all": {}})
        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, Optional[_LRScheduler]] = {}
        lr_ratio = max(1.0, self.environment.pipeline_config.get("lr_ratio", 1.0))
        default_lr = 1.0e-3 * lr_ratio
        for params_name, opt_setting in optimizers_settings.items():
            optimizer = opt_setting["optimizer"]
            optimizer_config = opt_setting["optimizer_config"]
            scheduler = opt_setting["scheduler"]
            scheduler_config = opt_setting["scheduler_config"]
            # we need to consider warmup here because it will modify the `lr`
            if scheduler != "warmup":
                optimizer_config.setdefault("lr", default_lr)
            else:
                multiplier = scheduler_config.setdefault("multiplier", 3)
                optimizer_config.setdefault("lr", default_lr / multiplier)
                default_max_warmup_step = int(round(3.0e5 / self.tr_loader.batch_size))
                num_step_per_epoch = self.state.num_step_per_epoch
                warmup_step = scheduler_config.setdefault(
                    "warmup_step",
                    min(default_max_warmup_step, 10 * num_step_per_epoch),
                )
                warmup_snapshot = int(warmup_step / self.state.num_step_per_snapshot)
                warmup_epoch = math.floor(warmup_step / num_step_per_epoch)
                self.state.plateau_start += warmup_snapshot
                self.state.min_epoch += warmup_epoch
            # the default settings of optimizers
            if optimizer == "nag":
                optimizer_config.setdefault("momentum", 0.999)
                optimizer_config.setdefault("weight_decay", 1e-7)
            if not isinstance(optimizer, str):
                optimizer_base = optimizer
            else:
                optimizer_base = optimizer_dict[optimizer]
            opt = self._define_optimizer(params_name, optimizer_base, optimizer_config)
            self.config["optimizer_config"] = optimizer_config
            self._optimizer_type = optimizer
            # scheduler
            default_lr_configs = self.default_lr_configs(opt, optimizer_config)
            default_lr_config = default_lr_configs.get(scheduler)
            if default_lr_config is not None:
                scheduler_config = update_dict(scheduler_config, default_lr_config)
            if scheduler == "warmup":
                sab = scheduler_config.get("scheduler_afterwards_base", "plateau")
                if sab == "warmup":
                    raise ValueError("warmup should not be used inside a warmup")
                sac = scheduler_config.get("scheduler_afterwards_config", {})
                default_lr_config = default_lr_configs.get(sab)
                sac = update_dict(sac, default_lr_config or {})
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
            if isinstance(scheduler, WarmupScheduler):
                scheduler = scheduler.scheduler_afterwards
            if scheduler is not None and scheduler_requires_metric(scheduler):
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
            loss_name = self.environment.model_config["loss"]
            if loss_name in Metrics.sign_dict and loss_name not in metric_types:
                metric_types = [loss_name]
                if loss_name == "quantile":
                    loss_config = self.environment.model_config["loss_config"]
                    metric_config["quantile_config"] = loss_config
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
                metric = Metrics(metric_type, sub_metric_config)
                self.metrics[metric_type] = metric
                metric.trainer = self
            if not 0.0 < metric_decay < 1.0:
                self.metrics_decay = None
            else:
                assert isinstance(self.metrics_decay, dict)
                self.metrics_decay[metric_type] = ScalarEMA(metric_decay)
        self.metrics_weights = metric_config.setdefault("weights", {})
        for metric_type in metric_types:
            self.metrics_weights.setdefault(metric_type, 1.0)

    @property
    def ddp(self) -> bool:
        return self.environment.ddp

    @property
    def model_for_training(self) -> torch.nn.Module:
        return self.ddp_model or self.model

    @property
    def validation_loader(self) -> PrefetchLoader:
        if self.cv_loader is None:
            if self.tr_loader_copy is None:
                raise ValueError("`tr_loader_copy` is not yet generated")
            return self.tr_loader_copy
        return self.cv_loader

    @property
    def validation_loader_name(self) -> str:
        loader = self.validation_loader
        return "tr" if loader is self.tr_loader_copy else "cv"

    @property
    def binary_threshold_loader(self) -> PrefetchLoader:
        if self.cv_loader is not None and len(self.cv_loader.data) >= 1000:
            return self.cv_loader
        if self.tr_loader_copy is None:
            raise ValueError("`tr_loader_copy` is not yet generated")
        return self.tr_loader_copy

    @property
    def binary_threshold_loader_name(self) -> str:
        loader = self.binary_threshold_loader
        return "tr" if loader is self.tr_loader_copy else "cv"

    # core

    @property
    def use_tqdm_in_cv(self) -> bool:
        if self.tqdm_settings.in_distributed:
            return False
        return self.tqdm_settings.use_tqdm_in_cv or self.state.is_terminate

    def _clip_norm_step(self) -> None:
        for opt in self.optimizers.values():
            self.grad_scaler.unscale_(opt)
        self._gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model_for_training.parameters(),
            max_norm=self.clip_norm,
        )

    def _optimizer_step(self) -> None:
        for opt in self.optimizers.values():
            self.grad_scaler.step(opt)
            self.grad_scaler.update()
        for param in self.model_for_training.parameters():
            param.grad = None

    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        kwargs = {}
        should_log_lr = self.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in self.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if self.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = self.intermediate.final_score
            should_log_lr = self.intermediate_updated
            self.intermediate_updated = False
        return should_log_lr, kwargs

    def _scheduler_step(self) -> None:
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                should_log_lr, kwargs = self._get_scheduler_settings(key, scheduler)
                if self.mlflow_client is not None and should_log_lr:
                    self.mlflow_client.log_metric(
                        self.run_id,
                        f"lr-{key}",
                        scheduler.get_last_lr()[0],  # type: ignore
                        step=self.state.step,
                    )
                if kwargs is None:
                    continue
                scheduler.step(**shallow_copy_dict(kwargs))  # type: ignore

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

    def _generate_binary_threshold(self) -> Optional[InferenceOutputs]:
        if not self.inference.need_binary_threshold:
            return None
        return self.inference.generate_binary_threshold(
            self.binary_threshold_loader,
            self.binary_threshold_loader_name,
            return_loss=self._metrics_need_loss,
            use_tqdm=self.use_tqdm_in_cv,
        )

    # return whether we need to terminate
    def _monitor_step(self) -> MonitorResults:
        outputs = None
        terminate = False
        if self.state.should_monitor:

            with timing_context(self, "monitor.binary_threshold", enable=self.timing):
                binary_outputs = None
                if self.update_bt_runtime and self.state.should_start_snapshot:
                    binary_outputs = self._generate_binary_threshold()

            with timing_context(self, "monitor.get_metrics", enable=self.timing):
                pack = self.get_metrics(binary_outputs=binary_outputs)
                outputs, self.intermediate = pack
                self.intermediate_updated = True
                if self.state.should_start_monitor_plateau:
                    if not self._monitor.plateau_flag:
                        self.log_msg(  # type: ignore
                            "start monitoring plateau",
                            self.info_prefix,
                            3,
                        )
                    self._monitor.plateau_flag = True

            with timing_context(self, "monitor.logging", enable=self.timing):
                if self.state.should_log_artifacts:
                    self._log_artifacts()
                if self.state.should_log_metrics_msg:
                    self._log_metrics_msg(self.intermediate)

            if self.state.should_start_snapshot:
                timing_name = "monitor.prune_trial"
                with timing_context(self, timing_name, enable=self.timing):
                    score = self.intermediate.final_score
                    if self.trial is not None:
                        self.trial.report(score, step=self.state.step)
                        if self.trial.should_prune():
                            raise optuna.TrialPruned()
                timing_name = "monitor.check_terminate"
                with timing_context(self, timing_name, enable=self.timing):
                    if self._monitor.check_terminate(score):
                        terminate = True

        return MonitorResults(terminate, outputs)

    def on_save_checkpoint(self, score: float) -> None:
        self.save_checkpoint(score)

    def _finalize(self, step_outputs: StepOutputs) -> None:
        if self.model.use_ema:
            with timing_context(self, "EMA", enable=self.timing):
                self.model.apply_ema()
        if self.state.should_log_losses:
            tr_losses = {f"tr_{k}": v for k, v in step_outputs.loss_items.items()}
            self._log_scalars(tr_losses)

    # core step on each epoch
    def _step(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        batch_indices: Optional[torch.Tensor],
    ) -> StepOutputs:
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            step_outputs = self.model.step(
                self.state,
                batch_idx,
                batch,
                batch_indices,
                "tr",
            )
        with timing_context(self, "loss.backward", enable=self.timing):
            loss = step_outputs.loss_dict["loss"]
            self.grad_scaler.scale(loss).backward()
        if self.clip_norm > 0.0:
            with timing_context(self, "clip_norm_step", enable=self.timing):
                self._clip_norm_step()
        with timing_context(self, "optimizer_step", enable=self.timing):
            self._optimizer_step()
        with timing_context(self, "scheduler_step", enable=self.timing):
            self._scheduler_step()
        self._finalize(step_outputs)
        return step_outputs

    # api

    def fit(
        self,
        tr_loader: DataLoaderProtocol,
        tr_loader_copy: DataLoaderProtocol,
        cv_loader: Optional[DataLoaderProtocol],
        tr_weights: Optional[np.ndarray],
        cv_weights: Optional[np.ndarray],
        *,
        enable_prefetch: bool = True,
    ) -> None:
        self.tr_loader = PrefetchLoader(
            tr_loader,
            self.device,
            enable_prefetch=enable_prefetch,
        )
        self.tr_loader_copy = PrefetchLoader(
            tr_loader_copy,
            self.device,
            enable_prefetch=enable_prefetch,
        )
        self.cv_loader: Optional[PrefetchLoader]
        if cv_loader is None:
            self.cv_loader = None
        else:
            self.cv_loader = PrefetchLoader(
                cv_loader,
                self.device,
                enable_prefetch=enable_prefetch,
            )
        self.state.inject_loader(tr_loader)
        # sample weights
        tr_weights_ = None if tr_weights is None else to_torch(tr_weights)
        cv_weights_ = None if cv_weights is None else to_torch(cv_weights)
        self.tr_weights, self.cv_weights = tr_weights_, cv_weights_
        # ddp
        self._init_ddp()
        # optimizer
        self._init_optimizers()
        # metrics
        self._init_metrics()
        # monitor
        monitor_config = self.config.setdefault("monitor_config", {})
        default_patience = max(4, math.ceil(math.log10(tr_loader.num_samples)))
        monitor_config.setdefault("patience", default_patience)
        self._monitor = TrainMonitor.monitor(self, **monitor_config)
        # train
        self.model.info()
        show_summary = self.show_summary
        if show_summary is None:
            show_summary = not self.tqdm_settings.in_distributed
        if show_summary and not self.is_loading:
            next_item = next(iter(self.tr_loader_copy))
            if self.tr_loader_copy.return_indices:
                assert isinstance(next_item, tuple)
                sample_batch = next_item[0]
            else:
                assert isinstance(next_item, dict)
                sample_batch = next_item
            summary(self.model, sample_batch)
        self._prepare_log()
        step_tqdm = None
        self._epoch_tqdm: Optional[tqdm] = None
        if self.tqdm_settings.use_tqdm:
            self._epoch_tqdm = tqdm(
                list(range(self.state.num_epoch)),
                position=self.tqdm_settings.position,
                desc=self.tqdm_settings.desc,
                leave=False,
            )
        has_ckpt = terminate = False
        while self.state.should_train:
            try:
                self.state.epoch += 1
                step_iterator = self.tr_loader
                if self.tqdm_settings.use_step_tqdm:
                    step_tqdm = step_iterator = tqdm(
                        step_iterator,
                        total=len(self.tr_loader),
                        position=self.tqdm_settings.position + 1,
                        leave=False,
                    )
                if self.ddp:
                    dist.barrier()
                for i, (batch, batch_indices) in enumerate(step_iterator):
                    self.state.step += 1
                    step_outputs = self._step(i, batch, batch_indices)
                    self.callback.after_step(step_outputs)
                    monitor_results = self._monitor_step()
                    self.callback.after_monitor(monitor_results)
                    terminate = monitor_results.terminate
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
                break
            if self.use_tqdm:
                assert self._epoch_tqdm is not None
                self._epoch_tqdm.total = self.state.num_epoch
                self._epoch_tqdm.update()
        if self.use_tqdm:
            if step_tqdm is not None:
                step_tqdm.close()
            assert self._epoch_tqdm is not None
            self._epoch_tqdm.close()
        # restore
        if os.path.isdir(self.checkpoint_folder):
            if not self.ddp:
                self.log_msg(  # type: ignore
                    "rolling back to the best checkpoint",
                    self.info_prefix,
                    3,
                )
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        outputs = self._generate_binary_threshold()
        _, self.final_results = self.get_metrics(binary_outputs=outputs)
        self._log_metrics_msg(self.final_results)
        if not has_ckpt:
            self.save_checkpoint(self.final_results.final_score)

    def get_metrics(
        self,
        *,
        binary_outputs: Optional[InferenceOutputs] = None,
        loader: Optional[PrefetchLoader] = None,
        loader_name: Optional[str] = None,
        metrics_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[InferenceOutputs, IntermediateResults]:
        is_custom_loader = loader is not None
        if binary_outputs is not None:
            outputs = binary_outputs
            probabilities = outputs.probabilities
            if not self.model.output_probabilities:
                logits = None
            else:
                logits = probabilities
            assert probabilities is not None
            outputs.results["predictions"] = self.inference.predict_with(probabilities)
        else:
            if not is_custom_loader:
                loader = self.validation_loader
                loader_name = self.validation_loader_name
            assert loader is not None
            outputs = self.inference.get_outputs(
                loader,
                loader_name,
                use_tqdm=self.use_tqdm_in_cv,
                return_loss=self._metrics_need_loss,
                getting_metrics=True,
                state=self.state,
            )
            results = self.inference.predict_from_outputs(
                outputs,
                return_all=True,
                requires_recover=False,
                returns_probabilities=False,
            )
            probabilities = None
            outputs.results.update(results)
            logits = outputs.results.get("logits")
        labels = outputs.labels
        results = outputs.results
        use_decayed = False
        signs: Dict[str, int] = {}
        metrics: Dict[str, float] = {}
        decayed_metrics: Dict[str, float] = {}
        for metric_type, metric_ins in self.metrics.items():
            if metric_ins is None:
                assert outputs.loss_items is not None
                signs[metric_type] = -1
                sub_metric = metrics[metric_type] = outputs.loss_items[metric_type]
            else:
                signs[metric_type] = metric_ins.sign
                if self.tr_loader.data.is_reg:
                    if metric_type == "quantile":
                        metric_key = "quantiles"
                    else:
                        metric_key = "predictions"
                    metric_predictions = results[metric_key]
                else:
                    if not metric_ins.requires_prob:
                        metric_predictions = results["predictions"]
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
                metric_kwargs = (metrics_kwargs or {}).get(metric_type)
                sub_metric = metric_ins.metric(
                    labels,
                    metric_predictions,
                    **shallow_copy_dict(metric_kwargs or {}),
                )
                metrics[metric_type] = float(sub_metric)
            if self.metrics_decay is not None and self.state.should_start_snapshot:
                use_decayed = True
                decayed = self.metrics_decay[metric_type].update("metric", sub_metric)
                decayed_metrics[metric_type] = decayed
        metrics_for_scoring = decayed_metrics if use_decayed else metrics
        if not is_custom_loader:
            if self._epoch_tqdm is not None:
                self._epoch_tqdm.set_postfix(metrics_for_scoring)
            self._log_scalars(metrics_for_scoring)
        weighted_metrics = {
            k: float(v * self.metrics_weights[k])
            for k, v in metrics_for_scoring.items()
        }
        weighted_scores = {k: v * signs[k] for k, v in weighted_metrics.items()}
        return outputs, IntermediateResults(
            metrics,
            weighted_metrics,
            weighted_scores,
            use_decayed,
            decayed_metrics,
        )

    def save_checkpoint(self, score: float, folder: Optional[str] = None) -> None:
        if folder is None:
            folder = self.checkpoint_folder
        # leave top_k snapshots only
        if self.state.max_snapshot_file > 0:
            checkpoints = self.model.sorted_checkpoints(folder)
            if len(checkpoints) >= self.state.max_snapshot_file:
                for file in checkpoints[self.state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        file = f"{self.model.pt_prefix}{self.state.epoch}.pt"
        torch.save(self.model.state_dict(), os.path.join(folder, file))
        # scores
        self.checkpoint_scores[file] = score
        with open(os.path.join(folder, self.model.scores_file), "w") as f:
            json.dump(self.checkpoint_scores, f)

    def restore_checkpoint(
        self,
        folder: str = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if folder is None:
            folder = self.checkpoint_folder
        return self.model.restore_checkpoint(folder, strict, state_dict_callback)


__all__ = [
    "IntermediateResults",
    "MonitoredMixin",
    "TrainMonitor",
    "MonitorResults",
    "TrainerCallback",
    "Trainer",
]
