from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from ...trainer import Trainer
from ...trainer import TqdmSettings
from ...trainer import OptimizerPack
from ...trainer import TrainerMonitor
from ...trainer import TrainerCallback
from ...protocol import MetricProtocol
from ...misc.internal_ import BasicMonitor
from ...misc.internal_ import MultipleMetrics
from ...misc.internal_.callbacks.general import _LogMetricsMsgCallback


def make_trainer(
    state_config: Optional[Dict[str, Any]] = None,
    *,
    workplace: str,
    num_epoch: int = 40,
    max_epoch: int = 1000,
    fixed_epoch: Optional[int] = None,
    fixed_steps: Optional[int] = None,
    log_steps: Optional[int] = None,
    valid_portion: float = 1.0,
    amp: bool = False,
    clip_norm: float = 0.0,
    metric_names: Optional[Union[str, List[str]]] = None,
    metric_configs: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    use_losses_as_metrics: Optional[bool] = None,
    loss_metrics_weights: Optional[Dict[str, float]] = None,
    monitor_names: Optional[Union[str, List[str]]] = None,
    monitor_configs: Optional[Dict[str, Any]] = None,
    callback_names: Optional[Union[str, List[str]]] = None,
    callback_configs: Optional[Dict[str, Any]] = None,
    lr: Optional[float] = None,
    optimizer_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    data_info_name: str = "data_info",
    metrics_log_file: str = "metrics.txt",
    finetune_config: Optional[Dict[str, Any]] = None,
    tqdm_settings: Optional[Dict[str, Any]] = None,
) -> Trainer:
    if state_config is None:
        state_config = {}
    if log_steps is not None:
        state_config.setdefault("num_step_per_log", log_steps)
        state_config.setdefault("snapshot_start_step", log_steps)
        state_config.setdefault("num_step_per_snapshot", log_steps)
    # metrics
    metrics: Optional[Union[MetricProtocol, MultipleMetrics]]
    if metric_names is None:
        metrics = None
    else:
        _metrics = MetricProtocol.make_multiple(metric_names, metric_configs)
        if isinstance(_metrics, MetricProtocol):
            metrics = _metrics
        else:
            metrics = MultipleMetrics(_metrics, weights=metric_weights)
    # monitor
    monitors: Optional[List[TrainerMonitor]]
    if monitor_names is None:
        monitors = None
    else:
        monitors = TrainerMonitor.make_multiple(monitor_names, monitor_configs)
    # callback
    callbacks: Optional[List[TrainerCallback]]
    if callback_names is None:
        callbacks = None
    else:
        callbacks = TrainerCallback.make_multiple(callback_names, callback_configs)
    # optimizer
    if optimizer_settings is None:
        optimizer_packs = None
    else:
        optimizer_packs = []
        for key, settings in optimizer_settings.items():
            optimizer = settings.get("optimizer")
            if optimizer is None:
                raise ValueError(f"optimizer must be provided (key={key}")
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
    if tqdm_settings is None:
        tqdm_settings = {}
    use_tqdm = tqdm_settings.setdefault("use_tqdm", False)
    # epoch
    if fixed_epoch is not None:
        num_epoch = max_epoch = fixed_epoch
    if max_epoch < num_epoch:
        raise ValueError("`max_epoch` should not be smaller than `num_epoch`")
    # default behaviors
    if monitors is None:
        monitors = [BasicMonitor()]
    if callbacks is None and not use_tqdm:
        callbacks = [_LogMetricsMsgCallback()]
    return Trainer(
        state_config,
        num_epoch=num_epoch,
        max_epoch=max_epoch,
        fixed_steps=fixed_steps,
        valid_portion=valid_portion,
        amp=amp,
        clip_norm=clip_norm,
        metrics=metrics,
        use_losses_as_metrics=use_losses_as_metrics,
        loss_metrics_weights=loss_metrics_weights,
        monitors=monitors,
        callbacks=callbacks,
        lr=lr,
        optimizer_name=optimizer_name,
        scheduler_name=scheduler_name,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        optimizer_packs=optimizer_packs,
        workplace=workplace,
        data_info_name=data_info_name,
        metrics_log_file=metrics_log_file,
        finetune_config=finetune_config,
        tqdm_settings=TqdmSettings(
            use_tqdm,
            tqdm_settings.setdefault("use_step_tqdm", use_tqdm),
            tqdm_settings.setdefault("use_tqdm_in_validation", False),
            tqdm_settings.setdefault("in_distributed", False),
            tqdm_settings.setdefault("tqdm_position", 0),
            tqdm_settings.setdefault("tqdm_desc", "epoch"),
        ),
    )
