import os
import json
import math
import torch
import shutil

from torch import nn
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
from dataclasses import dataclass
from torch.optim import Optimizer
from cftool.misc import update_dict
from cftool.misc import print_warning
from cftool.misc import prepare_workspace_from
from cftool.misc import truncate_string_to_length
from cftool.misc import Serializer
from cftool.misc import DataClassBase
from torch.optim.lr_scheduler import _LRScheduler

from .utils import TryLoadBlock
from .utils import InjectDefaultsMixin
from ..common import Block
from ...schema import device_type
from ...schema import trainer_callbacks
from ...schema import IData
from ...schema import IMetric
from ...schema import IDLModel
from ...schema import DLConfig
from ...schema import ITrainer
from ...schema import IInference
from ...schema import OptimizerPack
from ...schema import TrainerMonitor
from ...schema import TrainerCallback
from ...losses import losses
from ...models import DLEnsembleModel
from ...toolkit import _get_environ_workspace
from ...toolkit import scheduler_requires_metric
from ...trainer import get_scores
from ...trainer import get_sorted_checkpoints
from ...trainer import Trainer
from ...monitors import BasicMonitor
from ...callbacks import _LogMetricsMsgCallback
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import CHECKPOINTS_FOLDER
from ...optimizers import optimizer_dict
from ...schedulers import scheduler_dict
from ...schedulers import WarmupScheduler


# static blocks


@Block.register("set_defaults")
class SetDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        loss_name = config.loss_name
        module_name = config.module_name
        state_config = config.state_config
        callback_names = config.callback_names
        if loss_name is None:
            if losses.has(module_name):
                loss_name = module_name
                self._defaults["loss_name"] = loss_name
        if state_config is None:
            state_config = {}
        if "max_snapshot_file" not in state_config:
            state_config["max_snapshot_file"] = 25
            self._defaults["max_snapshot_file"] = 25
        if callback_names is None:
            if module_name in trainer_callbacks:
                callback_names = module_name
                self._defaults["callback_names"] = callback_names
        environ_workspace = _get_environ_workspace()
        if environ_workspace:
            config.workspace = environ_workspace
        config.loss_name = loss_name
        config.module_name = module_name
        config.state_config = state_config
        config.callback_names = callback_names
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
        # tqdm settings
        tqdm_settings = config.tqdm_settings
        if tqdm_settings is None:
            tqdm_settings = {}
        use_tqdm = tqdm_settings.setdefault("use_tqdm", False)
        tqdm_settings.setdefault("use_step_tqdm", use_tqdm)
        tqdm_settings.setdefault("use_tqdm_in_validation", False)
        tqdm_settings.setdefault("in_distributed", False)
        tqdm_settings.setdefault("tqdm_position", 0)
        tqdm_settings.setdefault("tqdm_desc", "epoch")
        config.tqdm_settings = tqdm_settings


@Block.register("prepare_workspace")
class PrepareWorkplaceBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        if not self.is_local_rank_0 or self.training_workspace is None:
            return
        if config.create_sub_workspace:
            workspace = prepare_workspace_from(self.training_workspace)
            config.workspace = workspace
            self._defaults["workspace"] = workspace


@dataclass
class StateInfo(DataClassBase):
    batch_size: int
    num_batches: int
    num_samples: int
    snapshot_start_step: int
    num_step_per_snapshot: int


@Block.register("extract_state_info")
class ExtractStateInfoBlock(TryLoadBlock):
    config: DLConfig
    state_info: StateInfo

    def try_load(self, folder: str) -> bool:
        info = Serializer.try_load_info(folder)
        if info is None:
            return False
        self.state_info = StateInfo(**info)
        return True

    def from_scratch(self, config: DLConfig) -> None:
        if self.data is None:
            raise ValueError(f"`data` should be provided for `ExtractStateInfoBlock`")
        # from loader
        loader = self.data.get_loaders()[0]
        batch_size = loader.batch_size
        num_batches = len(loader)
        num_samples = len(loader.dataset)
        # from config
        log_steps = config.log_steps
        state_config = config.state_config or {}
        # check log_steps
        if log_steps is not None:
            state_config.setdefault("num_step_per_log", log_steps)
            state_config.setdefault("snapshot_start_step", log_steps)
            state_config.setdefault("num_step_per_snapshot", log_steps)
        # check snapshot_start_step
        snapshot_start_step = state_config.get("snapshot_start_step")
        if snapshot_start_step is None:
            min_num_sample = state_config.get("min_num_sample", 3000)
            snapshot_start_step = math.ceil(min_num_sample / batch_size)
        # check num_step_per_snapshot
        num_step_per_snapshot = state_config.get("num_step_per_snapshot")
        if num_step_per_snapshot is None:
            num_snapshot_per_epoch = state_config.get("num_snapshot_per_epoch", 2)
            max_step_per_snapshot = state_config.get("max_step_per_snapshot", 1000)
            num_step_per_snapshot = max(1, int(num_batches / num_snapshot_per_epoch))
            num_step_per_snapshot = min(
                max_step_per_snapshot,
                num_step_per_snapshot,
            )
        # construct
        state_config["num_step_per_snapshot"] = num_step_per_snapshot
        state_config["snapshot_start_step"] = snapshot_start_step
        self.state_info = StateInfo(
            batch_size=batch_size,
            num_batches=num_batches,
            num_samples=num_samples,
            snapshot_start_step=snapshot_start_step,
            num_step_per_snapshot=num_step_per_snapshot,
        )
        config.state_config = state_config

    def dump_to(self, folder: str) -> None:
        if self.is_local_rank_0:
            Serializer.save_info(folder, info=self.state_info.asdict())


@Block.register("build_model")
class BuildModelBlock(InjectDefaultsMixin, Block):
    model: IDLModel

    def build(self, config: DLConfig) -> None:
        num_repeat = config.num_repeat
        m = IDLModel.from_config(config)
        if num_repeat is None:
            self.model = m
        else:
            self.model = DLEnsembleModel(m, num_repeat)


@Block.register("build_metrics")
class BuildMetricsBlock(Block):
    metrics: Optional[IMetric]

    def build(self, config: DLConfig) -> None:
        # build metrics
        metric_names = config.metric_names
        metric_configs = config.metric_configs
        metric_weights = config.metric_weights
        if metric_names is None:
            self.metrics = None
        else:
            self.metrics = IMetric.fuse(
                metric_names,
                metric_configs,
                metric_weights=metric_weights,
            )
        # check losses-as-metrics
        loss_metrics_weights = config.loss_metrics_weights
        use_losses_as_metrics = config.use_losses_as_metrics
        if self.metrics is None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            if not use_losses_as_metrics:
                msg = "`metrics` should be provided when not `use_losses_as_metrics`"
                raise ValueError(msg)
        if loss_metrics_weights is not None:
            if use_losses_as_metrics is None:
                use_losses_as_metrics = True
            elif not use_losses_as_metrics:
                raise ValueError(
                    "`use_losses_as_metrics` should not be False "
                    "when `loss_metrics_weights` is provided"
                )
        config.use_losses_as_metrics = use_losses_as_metrics


@Block.register("build_inference")
class BuildInferenceBlock(Block):
    inference: IInference

    def build(self, config: DLConfig) -> None:
        inference_type = config.inference_type
        inference_kw = dict(model=self.build_model.model)
        self.inference = IInference.make(inference_type, inference_kw)

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)


@Block.register("set_trainer_defaults")
class SetTrainerDefaultsBlock(InjectDefaultsMixin, Block):
    def build(self, config: DLConfig) -> None:
        model = self.build_model.model
        model.permute_trainer_config(config)
        # set some trainer defaults to deep learning tasks which work well in practice
        if config.monitor_names is None:
            config.monitor_names = "conservative"
            self._defaults["monitor_names"] = "conservative"
        module_name = config.module_name
        tqdm_settings = config.tqdm_settings
        callback_names = config.callback_names
        callback_configs = config.callback_configs
        if callback_names is None:
            callback_names = []
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        auto_callback = config.auto_callback
        if "mlflow" in callback_names and auto_callback:
            mlflow_config = callback_configs.setdefault("mlflow", {})
            if "experiment_name" not in mlflow_config:
                mlflow_config["experiment_name"] = module_name
                self._defaults["mlflow_experiment_name"] = module_name
        if "_log_metrics_msg" not in callback_names and auto_callback:
            self._defaults["additional_callbacks"] = ["_log_metrics_msg"]
            callback_names.insert(0, "_log_metrics_msg")
            verbose = False
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                verbose = True
            log_metrics_msg_config = callback_configs.setdefault("_log_metrics_msg", {})
            if "verbose" not in log_metrics_msg_config:
                log_metrics_msg_config["verbose"] = verbose
                self._defaults["log_metrics_msg_verbose"] = verbose
        config.tqdm_settings = tqdm_settings
        config.callback_names = callback_names
        config.callback_configs = callback_configs

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)


@Block.register("build_monitors")
class BuildMonitorsBlock(Block):
    monitors: List[TrainerMonitor]

    def build(self, config: DLConfig) -> None:
        monitor_names = config.monitor_names
        monitor_configs = config.monitor_configs
        if isinstance(monitor_names, str):
            monitor_names = [monitor_names]
        if monitor_names is None:
            self.monitors = [BasicMonitor()]
        else:
            self.monitors = TrainerMonitor.make_multiple(monitor_names, monitor_configs)


@Block.register("build_callbacks")
class BuildCallbacksBlock(Block):
    callbacks: List[TrainerCallback]

    def build(self, config: DLConfig) -> None:
        cb_names = config.callback_names
        cb_configs = config.callback_configs
        use_tqdm = (config.tqdm_settings or {}).get("use_tqdm", False)
        if cb_names is None:
            self.callbacks = [_LogMetricsMsgCallback(not use_tqdm)]
        else:
            self.callbacks = TrainerCallback.make_multiple(cb_names, cb_configs)
        for callback in self.callbacks:
            callback.initialize()


class DefaultOptimizerSettings(NamedTuple):
    lr: float = 1.0e-3
    optimizer_name: str = "adam"
    scheduler_name: Optional[str] = "warmup"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None

    def get_opt_pack(self, state_info: Optional[StateInfo]) -> OptimizerPack:
        optimizer_config = self.optimizer_config or {}
        scheduler_config = self.scheduler_config or {}
        if self.scheduler_name != "warmup":
            optimizer_config.setdefault("lr", self.lr)
        else:
            multiplier = scheduler_config.setdefault("multiplier", 3)
            optimizer_config.setdefault("lr", self.lr / multiplier)
            if state_info is None:
                scheduler_config.setdefault("warmup_step", 1000)
            else:
                default_max_warmup_step = int(round(3.0e5 / state_info.batch_size))
                scheduler_config.setdefault(
                    "warmup_step",
                    min(default_max_warmup_step, 10 * state_info.num_batches),
                )
        return OptimizerPack(
            "all",
            self.optimizer_name,
            self.scheduler_name,
            optimizer_config,
            scheduler_config,
        )

    def update_opt_pack(
        self,
        state_info: Optional[StateInfo],
        pack: OptimizerPack,
    ) -> OptimizerPack:
        self_pack = self.get_opt_pack(state_info)
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


@Block.register("build_optimizers")
class BuildOptimizersBlock(InjectDefaultsMixin, Block):
    config: DLConfig
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]
    schedulers_requires_metric: Set[str]

    def build(self, config: DLConfig) -> None:
        self.config = config
        state_info = self.extract_state_info.state_info
        # default settings
        settings: Dict[str, Any] = {}
        if config.lr is not None:
            settings["lr"] = config.lr
        if config.optimizer_name is not None:
            settings["optimizer_name"] = config.optimizer_name
        if config.scheduler_name is not None:
            if config.scheduler_name == "none":
                config.scheduler_name = None
            settings["scheduler_name"] = config.scheduler_name
        if config.optimizer_config is not None:
            settings["optimizer_config"] = config.optimizer_config
        if config.scheduler_config is not None:
            settings["scheduler_config"] = config.scheduler_config
        default_opt_settings = DefaultOptimizerSettings(**settings)
        self._defaults["default_optimizer_settings"] = default_opt_settings._asdict()
        # build
        optimizer_settings = config.optimizer_settings
        if optimizer_settings is None:
            optimizer_packs = [default_opt_settings.get_opt_pack(state_info)]
        else:
            optimizer_packs = []
            for scope, sub_settings in optimizer_settings.items():
                if sub_settings is None:
                    _, *defaults = default_opt_settings.get_opt_pack(state_info)
                    optimizer_packs.append(OptimizerPack(scope, *defaults))  # type: ignore
                    continue
                optimizer = sub_settings.get("optimizer")
                if optimizer is None:
                    raise ValueError(f"optimizer must be provided (scope={scope})")
                optimizer_packs.append(
                    OptimizerPack(
                        scope,
                        optimizer,
                        sub_settings.get("scheduler"),
                        sub_settings.get("optimizer_config"),
                        sub_settings.get("scheduler_config"),
                    )
                )
        # initialize
        self.optimizers = {}
        self.schedulers = {}
        for pack in optimizer_packs:
            pack = default_opt_settings.update_opt_pack(state_info, pack)
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

    @property
    def requirements(self) -> List[Type[Block]]:
        return [ExtractStateInfoBlock, BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def extract_state_info(self) -> ExtractStateInfoBlock:
        return self.get_previous(ExtractStateInfoBlock)

    def default_lr_configs(
        self,
        optimizer: Optimizer,
        optimizer_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        state_info = self.extract_state_info.state_info
        opt_lr = optimizer_config["lr"]
        # step
        step_default_cfg = {"step_size": 10 * state_info.num_batches}
        # exponential
        exp_gamma = (0.1**0.1) ** (1.0 / state_info.num_batches)
        exp_default_cfg = {"gamma": exp_gamma}
        # cyclic
        cyclic_default_cfg = {
            "base_lr": opt_lr,
            "max_lr": 1.0e-8,
            "step_size_up": 10 * state_info.num_batches,
            "gamma": exp_gamma,
        }
        if "momentum" not in optimizer.defaults:
            cyclic_default_cfg["cycle_momentum"] = False
        # cosine
        cosine_default_cfg = {
            "eta_min": 1.0e-8,
            "T_max": 10 * state_info.num_batches,
        }
        # cosine restarts
        cosine_restarts_default_cfg = {
            "eta_min": 1.0e-8,
            "T_0": 10 * state_info.num_batches,
        }
        # plateau
        plateau_default_cfg = {
            "mode": "max",
            "min_lr": 1.0e-8,
            "verbose": False,
            "patience": max(
                10 * state_info.num_step_per_snapshot,
                state_info.snapshot_start_step,
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
        model = self.build_model.model
        parameters: Any
        if pack.scope == "all":
            parameters = model.params_groups()
        else:
            attr = model
            scopes = pack.scope.split(".")
            for scope in scopes:
                new_attr = getattr(attr, scope, None)
                if new_attr is None:
                    raise ValueError(f"'{attr}' has no scope '{scope}'")
                attr = new_attr
            if not isinstance(attr, nn.Module):
                parameters = attr
            else:
                parameters = attr.parameters()
        optimizer_base = optimizer_dict[pack.optimizer_name]
        opt_config = pack.optimizer_config or {}
        opt = optimizer_base(parameters, **opt_config)
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


@Block.register("build_trainer")
class BuildTrainerBlock(Block):
    trainer: ITrainer

    def build(self, config: DLConfig) -> None:
        self.trainer = Trainer(config)


# runtime blocks


@Block.register("record_num_samples")
class RecordNumSamplesBlock(Block):
    def build(self, config: DLConfig) -> None:
        pass

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        _defaults["train_samples"] = len(data.train_dataset)
        if data.valid_dataset is None:
            _defaults["valid_samples"] = None
        else:
            _defaults["valid_samples"] = len(data.valid_dataset)
        _defaults.move_to_end("valid_samples", last=False)
        _defaults.move_to_end("train_samples", last=False)


@Block.register("report")
class ReportBlock(Block):
    config: DLConfig
    report_file = "report.txt"

    def build(self, config: DLConfig) -> None:
        self.config = config

    def run(self, data: IData, _defaults: OrderedDict, **kwargs: Any) -> None:
        if not self.is_local_rank_0 or self.training_workspace is None:
            return
        self._report_messages(
            "Internal Default Configurations Used by `carefree-learn`",
            _defaults,
            self.training_workspace,
        )
        original = self.config.__class__().asdict()
        external_configs: Dict[str, Any] = {}
        for k, v in self.config.asdict().items():
            if k in _defaults:
                continue
            ov = original[k]
            if v != ov:
                external_configs[k] = v
        self._report_messages(
            "External Configurations",
            external_configs,
            self.training_workspace,
        )

    def _report_messages(
        self,
        title: str,
        messages: Dict[str, Any],
        report_folder: str,
    ) -> None:
        def _stringify_item(
            item: Tuple[str, Any],
            prefix: Optional[str] = None,
            depth: int = 0,
        ) -> str:
            key, value = item
            if prefix is not None:
                key = f"{prefix}{key}"
            if not isinstance(value, dict) or not value or depth >= 2:
                key = truncate_string_to_length(key, span)
                return f"{key:>{span}s}   |   {value}"
            prefix = f"{key}."
            items = [
                _stringify_item((vk, vv), prefix, depth=depth + 1)
                for vk, vv in value.items()
            ]
            return "\n".join(items)

        span = 64
        length = 2 * span
        msg = "\n".join(
            [
                "=" * length,
                f"{title:^{length}s}",
                "-" * length,
                "\n".join(map(_stringify_item, messages.items())),
                "-" * length,
            ]
        )
        print(msg)
        if report_folder is not None:
            with open(os.path.join(report_folder, self.report_file), "a") as f:
                f.write(msg + "\n")


@Block.register("training")
class TrainingBlock(Block):
    trainer_config_file = "trainer_config.json"

    def build(self, config: DLConfig) -> None:
        pass

    @property
    def requirements(self) -> List[Type[Block]]:
        return [
            BuildModelBlock,
            BuildMetricsBlock,
            BuildInferenceBlock,
            BuildOptimizersBlock,
            BuildMonitorsBlock,
            BuildCallbacksBlock,
            BuildTrainerBlock,
        ]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_previous(BuildMetricsBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_previous(BuildInferenceBlock)

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    @property
    def build_monitors(self) -> BuildMonitorsBlock:
        return self.get_previous(BuildMonitorsBlock)

    @property
    def build_callbacks(self) -> BuildCallbacksBlock:
        return self.get_previous(BuildCallbacksBlock)

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_previous(BuildTrainerBlock)

    def run(
        self,
        data: IData,
        _defaults: OrderedDictType,
        *,
        device: device_type = None,
        **kwargs: Any,
    ) -> None:
        self.build_trainer.trainer.fit(
            data,
            self.build_model.model,
            self.build_metrics.metrics,
            self.build_inference.inference,
            self.build_optimizers.optimizers,
            self.build_optimizers.schedulers,
            self.build_monitors.monitors,
            self.build_callbacks.callbacks,
            self.build_optimizers.schedulers_requires_metric,
            config_export_file=self.trainer_config_file,
            device=device,
        )


# serialization blocks


@Block.register("serialize_data")
class SerializeDataBlock(Block):
    data: Optional[IData]
    config: DLConfig
    package_folder: str = "data_module"

    def build(self, config: DLConfig) -> None:
        self.data = None
        self.config = config

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        if self.training_workspace is not None:
            data_folder = os.path.join(self.training_workspace, self.package_folder)
            shutil.copytree(data_folder, folder)
        elif self.data is not None:
            Serializer.save(folder, self.data, save_npd=False)

    def load_from(self, folder: str) -> None:
        if os.path.isdir(folder):
            self.data = Serializer.load(folder, IData, load_npd=False)


@Block.register("serialize_model")
class SerializeModelBlock(Block):
    config: DLConfig

    verbose: bool = True
    ckpt_folder: Optional[str] = None
    ckpt_scores: Optional[Dict[str, float]] = None

    def build(self, config: DLConfig) -> None:
        self.config = config
        self.best_score = 0.0

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildModelBlock]

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_previous(BuildModelBlock)

    def save_extra(self, folder: str) -> None:
        if not self.is_local_rank_0:
            return
        warn_msg = "no checkpoints found at {}, current model states will be saved"
        if self.training_workspace is not None:
            ckpt_folder = os.path.join(self.training_workspace, CHECKPOINTS_FOLDER)
            if get_sorted_checkpoints(ckpt_folder):
                shutil.copytree(ckpt_folder, folder)
            else:
                if self.verbose:
                    print_warning(warn_msg.format(ckpt_folder))
                self._save_current(folder)
            return
        if self.ckpt_folder is None or self.ckpt_scores is None:
            if self.verbose:
                print_warning("current model states will be saved")
            self._save_current(folder)
        else:
            any_saved = False
            filtered_scores = {}
            os.makedirs(folder, exist_ok=True)
            for file, score in self.ckpt_scores.items():
                ckpt_path = os.path.join(self.ckpt_folder, file)
                if not os.path.isfile(ckpt_path):
                    if self.verbose:
                        msg = f"cannot find checkpoint at '{ckpt_path}', did you delete it?"
                        print_warning(msg)
                    continue
                any_saved = True
                filtered_scores[file] = score
                shutil.copyfile(ckpt_path, os.path.join(folder, file))
            if any_saved:
                with open(os.path.join(folder, SCORES_FILE), "w") as f:
                    json.dump(filtered_scores, f)
            else:
                if self.verbose:
                    print_warning(warn_msg.format(self.ckpt_folder))
                self._save_current(folder)

    def load_from(self, folder: str) -> None:
        model = self.build_model.model
        best_file = get_sorted_checkpoints(folder)[0]
        states = torch.load(os.path.join(folder, best_file))["states"]
        model.load_state_dict(states)
        scores = get_scores(folder)
        self.ckpt_folder = folder
        self.ckpt_scores = scores

    def _save_current(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        latest_file = f"{PT_PREFIX}-1.pt"
        latest_path = os.path.join(folder, latest_file)
        new_scores_path = os.path.join(folder, SCORES_FILE)
        self.build_model.model.save(latest_path)
        with open(new_scores_path, "w") as f:
            json.dump({latest_file: 0.0}, f)


@Block.register("serialize_optimizer")
class SerializeOptimizerBlock(Block):
    optimizer_file = "optimizers.pt"
    scheduler_file = "schedulers.pt"

    def build(self, config: DLConfig) -> None:
        pass

    @property
    def requirements(self) -> List[Type[Block]]:
        return [BuildOptimizersBlock]

    @property
    def build_optimizers(self) -> BuildOptimizersBlock:
        return self.get_previous(BuildOptimizersBlock)

    def save_extra(self, folder: str) -> None:
        optims = self.build_optimizers.optimizers
        scheds = self.build_optimizers.schedulers
        opt_d = {k: v.state_dict() for k, v in optims.items()}
        sch_d = {k: None if v is None else v.state_dict() for k, v in scheds.items()}
        os.makedirs(folder, exist_ok=True)
        torch.save(opt_d, os.path.join(folder, self.optimizer_file))
        torch.save(sch_d, os.path.join(folder, self.scheduler_file))

    def load_from(self, folder: str) -> None:
        optimizers = self.build_optimizers.optimizers
        schedulers = self.build_optimizers.schedulers
        opt_d = torch.load(os.path.join(folder, self.optimizer_file))
        sch_d = torch.load(os.path.join(folder, self.scheduler_file))
        for k, states in opt_d.items():
            optimizers[k].load_state_dict(states)
        for k, states in sch_d.items():
            k_sch = schedulers[k]
            if k_sch is not None:
                k_sch.load_state_dict(states)


__all__ = [
    "SetDefaultsBlock",
    "PrepareWorkplaceBlock",
    "ExtractStateInfoBlock",
    "BuildModelBlock",
    "BuildMetricsBlock",
    "BuildInferenceBlock",
    "SetTrainerDefaultsBlock",
    "BuildMonitorsBlock",
    "BuildCallbacksBlock",
    "BuildOptimizersBlock",
    "BuildTrainerBlock",
    "RecordNumSamplesBlock",
    "ReportBlock",
    "TrainingBlock",
    "SerializeDataBlock",
    "SerializeModelBlock",
    "SerializeOptimizerBlock",
]
