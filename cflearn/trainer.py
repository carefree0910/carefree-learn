import os
import re
import json
import math
import torch

import torch.distributed as dist

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from .types import tensor_dict_type
from .protocol import StepOutputs
from .protocol import LossProtocol
from .protocol import TrainerState
from .protocol import ModelProtocol
from .protocol import MetricsOutputs
from .protocol import MonitorResults
from .protocol import TrainerMonitor
from .protocol import MetricProtocol
from .protocol import InferenceProtocol
from .protocol import DataLoaderProtocol
from .protocol import ModelWithCustomSteps
from .constants import LOSS_KEY
from .constants import PT_PREFIX
from .constants import INFO_PREFIX
from .constants import SCORES_FILE
from .constants import ERROR_PREFIX
from .constants import WARNING_PREFIX
from .constants import CHECKPOINTS_FOLDER
from .misc.toolkit import summary
from .misc.toolkit import to_device
from .misc.toolkit import sort_dict_by_value
from .misc.toolkit import scheduler_requires_metric
from .misc.toolkit import eval_context
from .misc.toolkit import WithRegister
from .misc.internal_ import MultipleMetrics
from .misc.internal_ import ConservativeMonitor
from .modules.optimizers import optimizer_dict
from .modules.schedulers import scheduler_dict
from .modules.schedulers import WarmupScheduler


callback_dict: Dict[str, Type["TrainerCallback"]] = {}


class OptimizerPack(NamedTuple):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None


class DefaultOptimizerSettings(NamedTuple):
    lr: float = 1.0e-3
    optimizer_name: str = "adam"
    scheduler_name: Optional[str] = "warmup"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None

    def get_opt_pack(self, trainer: "Trainer") -> OptimizerPack:
        optimizer_config = self.optimizer_config or {}
        scheduler_config = self.scheduler_config or {}
        if self.scheduler_name != "warmup":
            optimizer_config.setdefault("lr", self.lr)
        else:
            multiplier = scheduler_config.setdefault("multiplier", 3)
            optimizer_config.setdefault("lr", self.lr / multiplier)
            batch_size = trainer.train_loader.batch_size
            default_max_warmup_step = int(round(3.0e5 / batch_size))
            num_step_per_epoch = trainer.state.num_step_per_epoch
            scheduler_config.setdefault(
                "warmup_step",
                min(default_max_warmup_step, 10 * num_step_per_epoch),
            )
        if self.optimizer_name == "nag":
            optimizer_config.setdefault("momentum", 0.999)
            optimizer_config.setdefault("weight_decay", 1e-7)
        return OptimizerPack(
            "all",
            self.optimizer_name,
            self.scheduler_name,
            optimizer_config,
            scheduler_config,
        )

    def update_opt_pack(self, trainer: "Trainer", pack: OptimizerPack) -> OptimizerPack:
        self_pack = self.get_opt_pack(trainer)
        opt_config = pack.optimizer_config or {}
        sch_config = pack.scheduler_config or {}
        if self_pack.optimizer_name != pack.optimizer_name:
            opt_config.setdefault("lr", self.lr)
        else:
            opt_config = update_dict(opt_config, self_pack.optimizer_config)
        if self_pack.scheduler_name == pack.scheduler_name:
            sch_config = update_dict(sch_config, self_pack.scheduler_config)
        return OptimizerPack(
            pack.scope,
            pack.optimizer_name,
            pack.scheduler_name,
            opt_config,
            sch_config,
        )


class DeviceInfo(NamedTuple):
    cuda: Optional[str]
    rank: Optional[int]

    @property
    def device(self) -> torch.device:
        if self.rank is not None:
            return torch.device(f"cuda:{self.rank}")
        return torch.device("cpu" if self.cuda is None else f"cuda:{self.cuda}")


class TrainerCallback(WithRegister):
    d: Dict[str, Type["TrainerCallback"]] = callback_dict
    is_rank_0: bool = True

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def initialize(self) -> None:
        pass

    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "Trainer",
    ) -> None:
        pass

    def mutate_train_loss_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "Trainer",
    ) -> None:
        pass

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        pass

    def log_metrics_msg(
        self,
        metric_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
    ) -> None:
        pass

    def log_artifacts(self, trainer: "Trainer") -> None:
        pass

    def after_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        pass

    def after_monitor(
        self,
        monitor_results: MonitorResults,
        state: TrainerState,
    ) -> None:
        pass

    def finalize(self, trainer: "Trainer") -> None:
        pass


class TqdmSettings(NamedTuple):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    in_distributed: bool = False
    position: int = 0
    desc: str = "epoch"


def get_sorted_checkpoints(checkpoint_folder: str) -> List[str]:
    # better checkpoints will be placed earlier,
    #  which means `checkpoints[0]` is the best checkpoint
    scores_path = os.path.join(checkpoint_folder, SCORES_FILE)
    if not os.path.isfile(scores_path):
        return []
    with open(scores_path, "r") as f:
        scores = json.load(f)
    return list(sort_dict_by_value(scores, reverse=True).keys())


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


class Trainer:
    loss: LossProtocol
    model: ModelProtocol
    metrics: Optional[MetricProtocol]
    monitors: List[TrainerMonitor]
    callbacks: List[TrainerCallback]
    state: TrainerState
    device_info: DeviceInfo
    train_loader: DataLoaderProtocol
    train_loader_copy: DataLoaderProtocol
    valid_loader: Optional[DataLoaderProtocol]
    inference: InferenceProtocol

    def __init__(
        self,
        state_config: Optional[Dict[str, Any]] = None,
        *,
        workplace: str,
        num_epoch: int = 40,
        max_epoch: int = 1000,
        fixed_steps: Optional[int] = None,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metrics: Optional[MetricProtocol] = None,
        loss_metrics_weights: Optional[Dict[str, float]] = None,
        monitors: Optional[Union[TrainerMonitor, List[TrainerMonitor]]] = None,
        callbacks: Optional[Union[TrainerCallback, List[TrainerCallback]]] = None,
        lr: Optional[float] = None,
        optimizer_name: Optional[str] = None,
        scheduler_name: Optional[str] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        optimizer_packs: Optional[Union[OptimizerPack, List[OptimizerPack]]] = None,
        metrics_log_file: str = "metrics.txt",
        ddp_config: Optional[Dict[str, Any]] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
        tqdm_settings: Optional[TqdmSettings] = None,
    ):
        self.tqdm_settings = tqdm_settings or TqdmSettings()
        self.state_config = state_config or {}
        self.max_epoch = max_epoch
        self.num_epoch = min(num_epoch, max_epoch)
        self.fixed_steps = fixed_steps
        self.valid_portion = valid_portion
        self.use_amp = amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.clip_norm = clip_norm
        if monitors is None:
            self.monitors = []
        else:
            if not isinstance(monitors, list):
                monitors = [monitors]
            self.monitors = monitors
        if callbacks is None:
            self.callbacks = []
        else:
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            self.callbacks = callbacks
        settings: Dict[str, Any] = {}
        if lr is not None:
            settings["lr"] = lr
        if optimizer_name is not None:
            settings["optimizer_name"] = optimizer_name
        if scheduler_name is not None:
            if scheduler_name == "none":
                scheduler_name = None
            settings["scheduler_name"] = scheduler_name
        if optimizer_config is not None:
            settings["optimizer_config"] = optimizer_config
        if scheduler_config is not None:
            settings["scheduler_config"] = scheduler_config
        self.default_opt_settings = DefaultOptimizerSettings(**settings)
        if optimizer_packs is None:
            self.optimizer_packs = None
        else:
            msg = None
            msg_fmt = "`{}` should not be provided when `optimizer_packs` is provided"
            if lr is not None:
                msg = msg_fmt.format("lr")
            elif optimizer_name is not None:
                msg = msg_fmt.format("optimizer_name")
            elif scheduler_name is not None:
                msg = msg_fmt.format("scheduler_name")
            elif optimizer_config is not None:
                msg = msg_fmt.format("optimizer_config")
            elif scheduler_config is not None:
                msg = msg_fmt.format("scheduler_config")
            if msg is not None:
                raise ValueError(msg)
            if not isinstance(optimizer_packs, list):
                optimizer_packs = [optimizer_packs]
            self.optimizer_packs = optimizer_packs
        self.metrics = metrics
        if metrics is not None:
            if not isinstance(metrics, MultipleMetrics):
                metrics.trainer = self
            else:
                for metric in metrics.metrics:
                    metric.trainer = self
        self.loss_metrics_weights = loss_metrics_weights
        if ddp_config is None:
            self.ddp = False
            self.rank = None
            self.ddp_config = {}
        else:
            self.ddp = True
            self.rank = ddp_config.get("rank")
            if self.rank is None:
                raise ValueError("`rank` should be provided when `ddp` is used")
            if ddp_config.get("world_size") is None:
                raise ValueError("`world_size` should be provided when `ddp` is used")
            self.ddp_config = shallow_copy_dict(ddp_config)
        self.is_rank_0 = not self.ddp or self.rank == 0
        for callback in self.callbacks:
            callback.is_rank_0 = self.is_rank_0
            callback.initialize()
        self.finetune_config = finetune_config
        # initialize artifact structure
        self.checkpoint_folder = None
        if self.is_rank_0:
            self.workplace = workplace
            os.makedirs(self.workplace, exist_ok=True)
            self.metrics_log_path = os.path.join(self.workplace, metrics_log_file)
            with open(self.metrics_log_path, "w"):
                pass
            self.checkpoint_folder = os.path.join(self.workplace, CHECKPOINTS_FOLDER)
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        # properties
        self.lr_metrics_updated = False
        self.intermediate: Optional[MetricsOutputs] = None
        self.final_results: Optional[MetricsOutputs] = None
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def device(self) -> torch.device:
        return self.device_info.device

    @property
    def use_tqdm_in_validation(self) -> bool:
        if not self.is_rank_0:
            return False
        if self.tqdm_settings.in_distributed:
            return False
        return self.tqdm_settings.use_tqdm_in_validation or self.state.is_terminate

    @property
    def validation_loader(self) -> DataLoaderProtocol:
        return self.valid_loader or self.train_loader_copy

    @property
    def input_sample(self) -> tensor_dict_type:
        return next(iter(self.train_loader_copy))

    @property
    def has_checkpoint_folder(self) -> bool:
        if self.checkpoint_folder is None:
            return False
        return os.path.isdir(self.checkpoint_folder)

    @property
    def model_has_custom_steps(self) -> bool:
        return isinstance(self.model, ModelWithCustomSteps)

    @property
    def model_for_training(self) -> torch.nn.Module:
        return self.ddp_model or self.model

    # init

    def _init_ddp(self) -> None:
        self.ddp_model = None
        if not self.ddp:
            return None
        assert self.rank is not None
        # monitor
        self.monitors = [] if not self.is_rank_0 else [ConservativeMonitor()]
        # ddp setup
        _setup_ddp(**self.ddp_config)
        self.ddp_model = DDP(self.model.to(self.rank), device_ids=[self.rank])

    def _init_finetune(self) -> None:
        if self.finetune_config is None:
            return None
        pretrained_ckpt = self.finetune_config.get("pretrained_ckpt")
        if pretrained_ckpt is None:
            raise ValueError("`rank` should be provided when `finetune` is triggered")
        print(f"{INFO_PREFIX}loading pretrained checkpoint from '{pretrained_ckpt}'...")
        d = torch.load(pretrained_ckpt, map_location=self.device)
        self.model.load_state_dict(d)
        freeze = self.finetune_config.get("freeze", "")
        freeze_except = self.finetune_config.get("freeze_except", "")
        if not freeze and not freeze_except:
            return None
        msg_fmt = f"{INFO_PREFIX}{'{}'} parameters will be {'{}'} under '{'{}'}'"
        param_names = []
        if freeze:
            num_frozen = 0
            for name, param in self.model.named_parameters():
                if re.match(freeze, name):
                    num_frozen += 1
                    param.requires_grad_(False)
                    param_names.append(name)
            msg = msg_fmt.format(num_frozen, "frozen", freeze)
        elif freeze_except:
            num_trainable = 0
            for name, param in self.model.named_parameters():
                if not re.match(freeze_except, name):
                    param.requires_grad_(False)
                else:
                    num_trainable += 1
                    param_names.append(name)
            msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
        else:
            msg = "`freeze` & `freeze_except` should not be provided simultaneously"
            raise ValueError(msg)
        print("\n".join(["=" * 100, msg, "-" * 100] + param_names + ["-" * 100]))

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
            "verbose": False,
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

    def _define_optimizer(self, pack: OptimizerPack) -> Optimizer:
        if pack.scope == "all":
            parameters = self.model_for_training.parameters()
        else:
            attr = getattr(self.model_for_training, pack.scope)
            if not isinstance(attr, torch.nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        optimizer_base = optimizer_dict[pack.optimizer_name]
        opt = optimizer_base(parameters, **(pack.optimizer_config or {}))
        self.optimizers[pack.scope] = opt
        return opt

    def _define_scheduler(self, optimizer: Optimizer, pack: OptimizerPack) -> None:
        if pack.scheduler_name is None:
            self.schedulers[pack.scope] = None
        else:
            scheduler = pack.scheduler_name
            opt_config = pack.optimizer_config or {}
            scheduler_config = pack.scheduler_config or {}
            default_lr_configs = self.default_lr_configs(optimizer, opt_config)
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
            scheduler_base = scheduler_dict[scheduler]
            self.schedulers[pack.scope] = scheduler_base(optimizer, **scheduler_config)

    def _init_optimizers(self) -> None:
        self.optimizers: Dict[str, Optimizer] = {}
        self.schedulers: Dict[str, Optional[_LRScheduler]] = {}
        # initialize
        if self.optimizer_packs is None:
            self.optimizer_packs = [self.default_opt_settings.get_opt_pack(self)]
        for pack in self.optimizer_packs:
            pack = self.default_opt_settings.update_opt_pack(self, pack)
            opt = self._define_optimizer(pack)
            self._define_scheduler(opt, pack)
        # check requires metric
        self.schedulers_requires_metric = set()
        for key, scheduler in self.schedulers.items():
            if scheduler is None:
                continue
            if isinstance(scheduler, WarmupScheduler):
                scheduler = scheduler.scheduler_afterwards
            if scheduler is not None and scheduler_requires_metric(scheduler):
                self.schedulers_requires_metric.add(key)

    # core

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
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs = {}
        should_log_lr = self.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in self.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if self.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = self.intermediate.final_score
            should_log_lr &= self.lr_metrics_updated
        return should_log_lr, kwargs

    def _scheduler_step(self) -> None:
        lr_metric_logged = False
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                should_log_lr, kwargs = self._get_scheduler_settings(key, scheduler)
                if should_log_lr:
                    lr_metric_logged = True
                    for callback in self.callbacks:
                        callback.log_lr(
                            f"lr-{key}",
                            scheduler.get_last_lr()[0],
                            self.state,
                        )
                scheduler.step(**shallow_copy_dict(kwargs))
        if lr_metric_logged:
            self.lr_metrics_updated = False

    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        if not self.is_rank_0:
            return None
        if self.epoch_tqdm is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.epoch_tqdm.set_postfix(metric_values)
        for callback in self.callbacks:
            callback.log_metrics(metrics_outputs, self.state)
        if self.state.should_log_artifacts:
            for callback in self.callbacks:
                callback.log_artifacts(self)
        if self.state.should_log_metrics_msg:
            for callback in self.callbacks:
                callback.log_metrics_msg(
                    metrics_outputs,
                    self.metrics_log_path,
                    self.state,
                )

    def _monitor_step(self) -> MonitorResults:
        terminate = False
        save_checkpoint = False
        if self.state.should_monitor:
            # get metrics
            self.intermediate = self.get_metrics(portion=self.valid_portion)
            self.lr_metrics_updated = True
            # logging
            self._logging_step(self.intermediate)
            # check terminate
            if self.state.should_start_snapshot:
                score = self.intermediate.final_score
                if any(monitor.snapshot(score) for monitor in self.monitors):
                    save_checkpoint = True
                if any(monitor.check_terminate(score) for monitor in self.monitors):
                    terminate = True
                for monitor in self.monitors:
                    monitor.handle_extension(self.state)
        return MonitorResults(terminate, save_checkpoint, self.intermediate)

    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        batch = to_device(batch, self.device)
        # kwargs
        forward_kwargs: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_forward_kwargs(forward_kwargs, self)
        loss_kwargs: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_loss_kwargs(loss_kwargs, self)
        # allow model defines its own training step
        if self.model_has_custom_steps and self.model.custom_train_step:
            return self.model.train_step(  # type: ignore
                batch_idx,
                batch,
                self,
                forward_kwargs,
                loss_kwargs,
            )
        # forward & loss
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            forward_results = self.model(batch_idx, batch, self.state, **forward_kwargs)
            loss_dict = self.loss(forward_results, batch, self.state, **loss_kwargs)
        # backward
        loss = loss_dict[LOSS_KEY]
        self.grad_scaler.scale(loss).backward()
        # clip norm
        if self.clip_norm > 0.0:
            self._clip_norm_step()
        # optimize
        self._optimizer_step()
        self._scheduler_step()
        return StepOutputs(forward_results, loss_dict)

    def _weighted_loss_score(self, loss_items: Dict[str, float]) -> float:
        if not self.loss_metrics_weights:
            return -loss_items[LOSS_KEY]
        score = 0.0
        for k, w in self.loss_metrics_weights.items():
            score -= loss_items[k] * w
        return score

    # api

    @property
    def configs(self) -> Dict[str, Any]:
        return {
            "state_config": self.state.configs,
            "valid_portion": self.valid_portion,
            "amp": self.use_amp,
            "clip_norm": self.clip_norm,
            "metrics": (
                None
                if self.metrics is None
                else self.metrics.__identifier__
                if not isinstance(self.metrics, MultipleMetrics)
                else [metric.__identifier__ for metric in self.metrics.metrics]
            ),
            "loss_metrics_weights": self.loss_metrics_weights,
            "monitors": [monitor.__identifier__ for monitor in self.monitors],
            "callbacks": [callback.__identifier__ for callback in self.callbacks],
            "optimizer_packs": (
                None
                if self.optimizer_packs is None
                else [pack._asdict() for pack in self.optimizer_packs]
            ),
            "ddp_config": self.ddp_config,
            "finetune_config": self.finetune_config,
            "tqdm_settings": self.tqdm_settings._asdict(),
            "device_info": self.device_info._asdict(),
        }

    def fit(
        self,
        loss: LossProtocol,
        model: ModelProtocol,
        inference: InferenceProtocol,
        train_loader: DataLoaderProtocol,
        valid_loader: Optional[DataLoaderProtocol] = None,
        *,
        configs_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        cuda: Optional[str] = None,
    ) -> "Trainer":
        self.device_info = DeviceInfo(cuda, self.rank)
        if self.is_rank_0:
            with open(os.path.join(self.workplace, "model.txt"), "w") as f:
                f.write(str(model))
        self.loss = loss.to(self.device)
        self.model = model.to(self.device)
        self.inference = inference
        self.train_loader = train_loader
        self.train_loader_copy = train_loader.copy()
        self.train_loader_copy.disable_shuffle()
        self.valid_loader = valid_loader
        self.state = TrainerState(
            train_loader,
            num_epoch=self.num_epoch,
            max_epoch=self.max_epoch,
            fixed_steps=self.fixed_steps,
            **self.state_config,
        )
        # ddp
        self._init_ddp()
        # optimizer
        self._init_optimizers()
        # callback
        self.model._init_with_trainer(self)
        # finetune
        self._init_finetune()
        # verbose
        if show_summary is None:
            show_summary = not self.tqdm_settings.in_distributed
        if self.is_rank_0:
            summary_msg = summary(
                self.model,
                to_device(self.input_sample, self.device),
                return_only=not show_summary,
            )
            with open(os.path.join(self.workplace, "summary.txt"), "w") as f:
                f.write(summary_msg)
        # tqdm
        step_tqdm = None
        self.epoch_tqdm: Optional[tqdm] = None
        if self.is_rank_0 and self.tqdm_settings.use_tqdm:
            self.epoch_tqdm = tqdm(
                list(range(self.state.num_epoch)),
                position=self.tqdm_settings.position,
                desc=self.tqdm_settings.desc,
                leave=False,
            )
        # train
        has_ckpt = terminate = False
        if self.epoch_tqdm is None:
            print(f"{INFO_PREFIX}entered training loop")
        if configs_export_file is not None:
            configs_export_path = os.path.join(self.workplace, configs_export_file)
            with open(configs_export_path, "w") as f:
                json.dump(self.configs, f)
        while self.state.should_train:
            try:
                self.state.epoch += 1
                step_iterator = self.train_loader
                if self.is_rank_0 and self.tqdm_settings.use_step_tqdm:
                    step_tqdm = step_iterator = tqdm(
                        step_iterator,
                        total=len(self.train_loader),
                        position=self.tqdm_settings.position + 1,
                        leave=False,
                    )
                for i, batch in enumerate(step_iterator):
                    self.state.step += 1
                    step_outputs = self._step(i, batch)
                    for callback in self.callbacks:
                        callback.after_step(step_outputs, self.state)
                    monitor_results = self._monitor_step()
                    for callback in self.callbacks:
                        callback.after_monitor(monitor_results, self.state)
                    if self.is_rank_0 and monitor_results.save_checkpoint:
                        metric_outputs = monitor_results.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                    terminate = monitor_results.terminate or self.state.should_terminate
                    if terminate:
                        break
            except KeyboardInterrupt:
                print(f"{ERROR_PREFIX}keyboard interrupted")
                terminate = True
            if terminate:
                break
            if self.epoch_tqdm is not None:
                self.epoch_tqdm.total = self.state.num_epoch
                self.epoch_tqdm.update()
        if self.epoch_tqdm is not None:
            if step_tqdm is not None:
                step_tqdm.close()
            self.epoch_tqdm.close()
        # restore
        if self.is_rank_0 and self.has_checkpoint_folder:
            if not self.tqdm_settings.in_distributed:
                print(f"{INFO_PREFIX}rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if self.is_rank_0:
            self.final_results = self.get_metrics(portion=self.valid_portion)
            self._logging_step(self.final_results)
            if not has_ckpt:
                self.save_checkpoint(self.final_results.final_score)
        for callback in self.callbacks:
            callback.finalize(self)
        return self

    def get_metrics(
        self,
        *,
        portion: float = 1.0,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        if loader is None:
            loader = self.validation_loader
        if self.model_has_custom_steps and self.model.custom_evaluate_step:
            use_grad = self.inference.use_grad_in_predict
            try:
                with eval_context(self.model, use_grad=use_grad):
                    rs = self.model.evaluate_step(loader, portion, self)  # type: ignore
            except:
                self.inference.use_grad_in_predict = True
                with eval_context(self.model, use_grad=True):
                    rs = self.model.evaluate_step(loader, portion, self)  # type: ignore
            return rs
        outputs = self.inference.get_outputs(
            loader,
            portion=portion,
            state=self.state,
            metrics=self.metrics,
            loss=self.loss if self.metrics is None else None,
            return_outputs=False,
        )
        if self.metrics is not None:
            assert outputs.metric_outputs is not None
            return outputs.metric_outputs
        loss_items = outputs.loss_items
        assert loss_items is not None
        score = self._weighted_loss_score(loss_items)
        return MetricsOutputs(score, loss_items)

    # checkpointing

    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        if folder is None:
            if self.checkpoint_folder is None:
                assert not self.is_rank_0
                msg = "`save_checkpoint` should not be called when not `is_rank_0`"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        # leave top_k snapshots only
        if self.state.max_snapshot_file > 0:
            checkpoints = get_sorted_checkpoints(folder)
            if len(checkpoints) >= self.state.max_snapshot_file:
                for file in checkpoints[self.state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        file = f"{PT_PREFIX}{self.state.step}.pt"
        torch.save(self.model.state_dict(), os.path.join(folder, file))
        # scores
        scores = {} if no_history else self.checkpoint_scores
        scores[file] = score
        with open(os.path.join(folder, SCORES_FILE), "w") as f:
            json.dump(sort_dict_by_value(scores, reverse=True), f)

    def restore_checkpoint(
        self,
        folder: str = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if folder is None:
            if self.checkpoint_folder is None:
                assert not self.is_rank_0
                msg = "`restore_checkpoint` should not be called when not `is_rank_0`"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            if not self.tqdm_settings.in_distributed:
                print(f"{WARNING_PREFIX}no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            if not self.tqdm_settings.in_distributed:
                print(f"{INFO_PREFIX}restoring from {model_file}")
            states = torch.load(model_file, map_location=self.device)
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success


__all__ = [
    "get_sorted_checkpoints",
    "Trainer",
    "TrainerCallback",
    "StepOutputs",
    "OptimizerPack",
]
