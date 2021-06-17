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
from ...misc.internal_ import MultipleMetrics


def make_trainer(
    state_config: Optional[Dict[str, Any]] = None,
    *,
    workplace: str,
    num_epoch: int = 40,
    max_epoch: int = 1000,
    fixed_epoch: Optional[int] = None,
    valid_portion: float = 1.0,
    amp: bool = False,
    clip_norm: float = 0.0,
    metric_names: Optional[Union[str, List[str]]] = None,
    metric_configs: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    loss_metrics_weights: Optional[Dict[str, float]] = None,
    monitor_names: Optional[Union[str, List[str]]] = None,
    monitor_configs: Optional[Dict[str, Any]] = None,
    callback_names: Optional[Union[str, List[str]]] = None,
    callback_configs: Optional[Dict[str, Any]] = None,
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    metrics_log_file: str = "metrics.txt",
    rank: Optional[int] = None,
    tqdm_settings: Optional[Dict[str, Any]] = None,
) -> Trainer:
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
    return Trainer(
        state_config,
        num_epoch=num_epoch,
        max_epoch=max_epoch,
        valid_portion=valid_portion,
        amp=amp,
        clip_norm=clip_norm,
        metrics=metrics,
        loss_metrics_weights=loss_metrics_weights,
        monitors=monitors,
        callbacks=callbacks,
        optimizer_packs=optimizer_packs,
        workplace=workplace,
        metrics_log_file=metrics_log_file,
        rank=rank,
        tqdm_settings=TqdmSettings(
            use_tqdm,
            tqdm_settings.setdefault("use_step_tqdm", use_tqdm),
            tqdm_settings.setdefault("use_tqdm_in_validation", False),
            tqdm_settings.setdefault("in_distributed", False),
            tqdm_settings.setdefault("tqdm_position", 0),
            tqdm_settings.setdefault("tqdm_desc", "epoch"),
        ),
    )
