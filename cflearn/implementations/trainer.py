from typing import List
from typing import Union
from typing import Optional

from .monitors import BasicMonitor
from .callbacks import _LogMetricsMsgCallback
from ..schema import TqdmSettings
from ..schema import OptimizerPack
from ..schema import TrainerMonitor
from ..schema import _IMetric
from ..schema import _MultipleMetrics
from ..schema import TrainerConfig
from ..schema import TrainerCallback
from ..trainer import Trainer


def make_trainer(config: TrainerConfig) -> Trainer:
    state_config = config.state_config
    log_steps = config.log_steps
    if state_config is None:
        state_config = {}
    if log_steps is not None:
        state_config.setdefault("num_step_per_log", log_steps)
        state_config.setdefault("snapshot_start_step", log_steps)
        state_config.setdefault("num_step_per_snapshot", log_steps)
    # metrics
    metric_names = config.metric_names
    metric_configs = config.metric_configs
    metric_weights = config.metric_weights
    metrics: Optional[Union[_IMetric, _MultipleMetrics]]
    if metric_names is None:
        metrics = None
    else:
        metrics = _IMetric.fuse(
            metric_names,
            metric_configs,
            metric_weights=metric_weights,
        )
    # monitor
    monitor_names = config.monitor_names
    monitor_configs = config.monitor_configs
    monitors: Optional[List[TrainerMonitor]]
    if monitor_names is None:
        monitors = None
    else:
        monitors = TrainerMonitor.make_multiple(monitor_names, monitor_configs)
    # callback
    callback_names = config.callback_names
    callback_configs = config.callback_configs
    callbacks: Optional[List[TrainerCallback]]
    if callback_names is None:
        callbacks = None
    else:
        callbacks = TrainerCallback.make_multiple(callback_names, callback_configs)
    # optimizer
    optimizer_settings = config.optimizer_settings
    if optimizer_settings is None:
        optimizer_packs = None
    else:
        optimizer_packs = []
        for key, settings in optimizer_settings.items():
            optimizer = settings.get("optimizer")
            if optimizer is None:
                raise ValueError(f"optimizer must be provided (key={key})")
            optimizer_packs.append(
                OptimizerPack(
                    key,
                    optimizer,
                    settings.get("scheduler"),
                    settings.get("optimizer_config"),
                    settings.get("scheduler_config"),
                )
            )
    # tqdm
    tqdm_settings = config.tqdm_settings
    if tqdm_settings is None:
        tqdm_settings = {}
    use_tqdm = tqdm_settings.setdefault("use_tqdm", False)
    # epoch
    max_epoch = config.max_epoch
    num_epoch = config.num_epoch
    fixed_epoch = config.fixed_epoch
    if fixed_epoch is not None:
        num_epoch = max_epoch = fixed_epoch
    if max_epoch < num_epoch:
        raise ValueError("`max_epoch` should not be smaller than `num_epoch`")
    # default behaviors
    if monitors is None:
        monitors = [BasicMonitor()]
    if callbacks is None:
        callbacks = [_LogMetricsMsgCallback(not use_tqdm)]
    return Trainer(
        state_config,
        num_epoch=num_epoch,
        max_epoch=max_epoch,
        fixed_steps=config.fixed_steps,
        valid_portion=config.valid_portion,
        amp=config.amp,
        clip_norm=config.clip_norm,
        metrics=metrics,
        use_losses_as_metrics=config.use_losses_as_metrics,
        loss_metrics_weights=config.loss_metrics_weights,
        recompute_train_losses_in_eval=config.recompute_train_losses_in_eval,
        monitors=monitors,
        callbacks=callbacks,
        lr=config.lr,
        optimizer_name=config.optimizer_name,
        scheduler_name=config.scheduler_name,
        optimizer_config=config.optimizer_config,
        scheduler_config=config.scheduler_config,
        update_scheduler_per_epoch=config.update_scheduler_per_epoch,
        optimizer_packs=optimizer_packs,
        use_zero=config.use_zero,
        workplace=config.workplace,
        data_info_name=config.data_info_name,
        metrics_log_file=config.metrics_log_file,
        finetune_config=config.finetune_config,
        tqdm_settings=TqdmSettings(
            use_tqdm,
            tqdm_settings.setdefault("use_step_tqdm", use_tqdm),
            tqdm_settings.setdefault("use_tqdm_in_validation", False),
            tqdm_settings.setdefault("in_distributed", False),
            tqdm_settings.setdefault("tqdm_position", 0),
            tqdm_settings.setdefault("tqdm_desc", "epoch"),
        ),
    )


__all__ = [
    "make_trainer",
]
