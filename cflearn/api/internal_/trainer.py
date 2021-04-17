from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

from ...trainer import callback_dict
from ...trainer import Trainer
from ...trainer import TqdmSettings
from ...trainer import OptimizerPack
from ...trainer import TrainerMonitor
from ...trainer import TrainerCallback
from ...protocol import metric_dict
from ...protocol import monitor_dict
from ...misc.internal_.metrics import MultipleMetrics


def make_trainer(
    state_config: Optional[Dict[str, Any]] = None,
    *,
    workplace: str,
    num_epoch: int = 40,
    valid_portion: float = 1.0,
    amp: bool = False,
    clip_norm: float = 0.0,
    metric_names: Optional[Union[str, List[str]]] = None,
    metric_configs: Optional[Dict[str, Any]] = None,
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
    if metric_names is None:
        metrics = None
    else:
        if metric_configs is None:
            metric_configs = {}
        if isinstance(metric_names, str):
            metrics = metric_dict[metric_names](**metric_configs)
        else:
            metrics = MultipleMetrics(
                [
                    metric_dict[name](**(metric_configs.get(name, {})))
                    for name in metric_names
                ]
            )
    # monitor
    monitors: Optional[List[TrainerMonitor]]
    if monitor_names is None:
        monitors = None
    else:
        if monitor_configs is None:
            monitor_configs = {}
        if isinstance(monitor_names, str):
            monitors = [monitor_dict[monitor_names](**monitor_configs)]
        else:
            monitors = [
                monitor_dict[name](**(monitor_configs.get(name, {})))
                for name in monitor_names
            ]
    # callback
    callbacks: Optional[List[TrainerCallback]]
    if callback_names is None:
        callbacks = None
    else:
        if callback_configs is None:
            callback_configs = {}
        if isinstance(callback_names, str):
            callbacks = [callback_dict[callback_names](**callback_configs)]
        else:
            callbacks = [
                callback_dict[name](**(callback_configs.get(name, {})))
                for name in callback_names
            ]
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
    return Trainer(
        state_config,
        num_epoch=num_epoch,
        valid_portion=valid_portion,
        amp=amp,
        clip_norm=clip_norm,
        metrics=metrics,
        monitors=monitors,
        callbacks=callbacks,
        optimizer_packs=optimizer_packs,
        workplace=workplace,
        metrics_log_file=metrics_log_file,
        rank=rank,
        tqdm_settings=TqdmSettings(
            use_tqdm,
            tqdm_settings.setdefault("use_step_tqdm", use_tqdm),
            tqdm_settings.setdefault("use_tqdm_in_cv", False),
            tqdm_settings.setdefault("in_distributed", False),
            tqdm_settings.setdefault("tqdm_position", 0),
            tqdm_settings.setdefault("tqdm_desc", "epoch"),
        ),
    )
