import os
import json
import torch
import inspect

from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import timestamp
from cftool.misc import update_dict
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin
from cfdata.tabular import task_type_type
from cfdata.tabular import TimeSeriesConfig
from optuna.trial import Trial

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from .types import general_config_type


configs_dict: Dict[str, Dict[str, Type["Configs"]]] = {}


class Configs(ABC, LoggingMixin):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.config = config

    @abstractmethod
    def get_default(self) -> Dict[str, Any]:
        pass

    def pop(self) -> Dict[str, Any]:
        default = self.get_default()
        if self.config is None:
            return default
        return update_dict(shallow_copy_dict(self.config), default)

    def setdefault(self, key: str, value: Any) -> Any:
        if self.config is None:
            self.config = {key: value}
            return value
        return self.config.setdefault(key, value)

    @classmethod
    def register(cls, scope: str, name: str) -> Callable[[Type], Type]:
        global configs_dict

        def before(cls_: Type) -> None:
            cls_.name = name

        return register_core(
            name,
            configs_dict.setdefault(scope, {}),
            before_register=before,
        )

    @classmethod
    def get(cls, scope: str, name: str, **kwargs: Any) -> "Configs":
        return configs_dict[scope][name](kwargs)


def _parse_config(config: general_config_type) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)
    return shallow_copy_dict(config)


class Elements(NamedTuple):
    model: str = "fcnn"
    use_amp: Optional[bool] = None
    use_simplify_data: Optional[bool] = None
    config: general_config_type = None
    increment_config: general_config_type = None
    delim: Optional[str] = None
    task_type: Optional[task_type_type] = None
    skip_first: Optional[bool] = None
    cv_split: Optional[Union[float, int]] = None
    min_epoch: Optional[int] = None
    num_epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    fixed_epoch: Optional[int] = None
    batch_size: Optional[int] = None
    max_snapshot_num: Optional[int] = None
    clip_norm: Optional[float] = None
    ema_decay: Optional[float] = None
    ts_config: Optional[TimeSeriesConfig] = None
    aggregation: Optional[str] = None
    aggregation_config: Optional[Dict[str, Any]] = None
    ts_label_collator_config: Optional[Dict[str, Any]] = None
    data_config: Optional[Dict[str, Any]] = None
    read_config: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Union[str, List[str]]] = None
    metric_config: Optional[Dict[str, Any]] = None
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    optimizers: Optional[Dict[str, Any]] = None
    logging_file: Optional[str] = None
    logging_folder: Optional[str] = None
    trigger_logging: Optional[bool] = None
    trial: Optional[Trial] = None
    tracker_config: Optional[Dict[str, Any]] = None
    cuda: Optional[Union[int, str]] = None
    verbose_level: Optional[int] = None
    use_timing_context: Optional[bool] = None
    use_tqdm: Optional[bool] = None
    extra_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> "Elements":
        spec = inspect.getfullargspec(cls).args[1:-1]
        main_configs = {key: kwargs.pop(key) for key in spec if key in kwargs}
        return cls(**main_configs, extra_config=kwargs)


class Environment:
    def __init__(self, config: Optional[Dict[str, Any]]):
        self.config = self.pop(config)

    def __getattr__(self, item: str) -> Any:
        return self.config[item]

    @property
    def device(self):
        cuda = self.cuda
        if cuda == "cpu":
            return torch.device("cpu")
        if cuda is not None:
            return torch.device(f"cuda:{cuda}")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def pipeline_config(self) -> Dict[str, Any]:
        return self.config

    @property
    def trainer_config(self) -> Dict[str, Any]:
        return self.config["trainer_config"]

    @property
    def model_config(self) -> Dict[str, Any]:
        return self.config["model_config"]

    @staticmethod
    def pop(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = config or {}
        # pipeline general
        model = kwargs.setdefault("model", "fcnn")
        kwargs.setdefault("model", "fcnn")
        kwargs.setdefault("use_tqdm", True)
        kwargs.setdefault("use_timing_context", True)
        kwargs.setdefault("use_binary_threshold", True)
        kwargs.setdefault("data_config", {})
        kwargs.setdefault("read_config", {})
        kwargs.setdefault("cv_split", None)
        kwargs.setdefault("min_cv_split", 100)
        kwargs.setdefault("max_cv_split", 10000)
        kwargs.setdefault("max_cv_split_ratio", 0.5)
        kwargs.setdefault("cv_split_order", "auto")
        kwargs.setdefault("binary_config", {})
        kwargs.setdefault("shuffle_tr", True)
        batch_size = kwargs.setdefault("batch_size", 128)
        kwargs.setdefault("cv_batch_size", 5 * batch_size)
        kwargs.setdefault("sampler_config", {})
        kwargs.setdefault("ts_label_collator_config", {})
        log_folder = kwargs.setdefault("logging_folder", os.path.join("_logs", model))
        log_file = kwargs.get("logging_file")
        if log_file is not None:
            log_path = os.path.join(log_folder, log_file)
        else:
            log_path = os.path.abspath(os.path.join(log_folder, f"{timestamp()}.log"))
        kwargs["_logging_path_"] = log_path
        kwargs.setdefault("trigger_logging", False)
        # pipeline -> misc
        kwargs.setdefault("cuda", None)
        kwargs.setdefault("trial", None)
        kwargs.setdefault("tracker_config", None)
        kwargs.setdefault("verbose_level", 2)
        # trainer general
        trainer_config = kwargs.setdefault("trainer_config", {})
        trainer_config.setdefault("update_binary_threshold_at_runtime", False)
        trainer_config.setdefault("clip_norm", 0.0)
        use_amp = trainer_config.get("use_amp", False)
        trainer_config["use_amp"] = use_amp and amp is not None
        default_checkpoint_folder = os.path.join(log_folder, "checkpoints")
        trainer_config.setdefault("checkpoint_folder", default_checkpoint_folder)
        # trainer -> optimizers
        optimizers_settings = trainer_config.setdefault("optimizers", {"all": {}})
        for params_name, opt_setting in optimizers_settings.items():
            opt_setting.setdefault("optimizer", "adam")
            optimizer_config = opt_setting.setdefault("optimizer_config", {})
            opt_setting.setdefault("scheduler", "plateau")
            opt_setting.setdefault("scheduler_config", {})
            optimizer_config.setdefault("lr", 1e-3)
        # trainer -> metrics
        metric_config = trainer_config.setdefault("metric_config", {})
        metric_config.setdefault("types", "auto")
        metric_config.setdefault("decay", 0.1)
        # model general
        model_config = kwargs.setdefault("model_config", {})
        model_config.setdefault("ema_decay", 0.0)
        encoding_methods = model_config.get("encoding_methods", {})
        encoding_configs = model_config.get("encoding_configs", {})
        encoding_methods = {str(k): v for k, v in encoding_methods.items()}
        encoding_configs = {str(k): v for k, v in encoding_configs.items()}
        model_config["encoding_methods"] = encoding_methods
        model_config["encoding_configs"] = encoding_configs
        model_config.setdefault("default_encoding_configs", {})
        model_config.setdefault("loss_config", {})
        return kwargs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Environment":
        return cls(config)

    @classmethod
    def from_json(cls, json_path: str) -> "Environment":
        with open(json_path, "r") as f:
            return cls(json.load(f))

    @classmethod
    def from_elements(cls, elements: Elements) -> "Environment":
        if elements.extra_config is None:
            kwargs = {}
        else:
            kwargs = shallow_copy_dict(elements.extra_config)
        cfg, inc_cfg = map(_parse_config, [elements.config, elements.increment_config])
        update_dict(update_dict(inc_cfg, cfg), kwargs)
        # pipeline general
        kwargs["model"] = elements.model
        if elements.cv_split is not None:
            kwargs["cv_split"] = elements.cv_split
        if elements.use_tqdm is not None:
            kwargs["use_tqdm"] = elements.use_tqdm
        if elements.use_timing_context is not None:
            kwargs["use_timing_context"] = elements.use_timing_context
        if elements.batch_size is not None:
            kwargs["batch_size"] = elements.batch_size
        if elements.ts_label_collator_config is not None:
            kwargs["ts_label_collator_config"] = elements.ts_label_collator_config
        data_config = elements.data_config or {}
        if elements.use_simplify_data is not None:
            data_config["simplify"] = elements.use_simplify_data
        if elements.ts_config is not None:
            data_config["time_series_config"] = elements.ts_config
        if elements.task_type is not None:
            data_config["task_type"] = elements.task_type
        read_config = elements.read_config or {}
        if elements.delim is not None:
            read_config["delim"] = elements.delim
        if elements.skip_first is not None:
            read_config["skip_first"] = elements.skip_first
        kwargs["data_config"] = data_config
        kwargs["read_config"] = read_config
        sampler_config = kwargs.setdefault("sampler_config", {})
        if elements.aggregation is not None:
            sampler_config["aggregation"] = elements.aggregation
        if elements.aggregation_config is not None:
            sampler_config["aggregation_config"] = elements.aggregation_config
        if elements.logging_folder is not None:
            if elements.logging_file is not None:
                logging_file = elements.logging_file
            else:
                logging_file = f"{elements.model}_{timestamp()}.log"
            kwargs["logging_folder"] = elements.logging_folder
            kwargs["logging_file"] = logging_file
        if elements.trigger_logging is not None:
            kwargs["trigger_logging"] = elements.trigger_logging
        # trainer general
        trainer_config = kwargs.setdefault("trainer_config", {})
        if elements.use_amp is not None:
            trainer_config["use_amp"] = elements.use_amp
        min_epoch = elements.min_epoch
        num_epoch = elements.num_epoch
        max_epoch = elements.max_epoch
        if elements.fixed_epoch is not None:
            msg = "`{}` should not be provided when `fixed_epoch` is provided"
            if min_epoch is not None:
                raise ValueError(msg.format("min_epoch"))
            if num_epoch is not None:
                raise ValueError(msg.format("num_epoch"))
            if max_epoch is not None:
                raise ValueError(msg.format("max_epoch"))
            min_epoch = num_epoch = max_epoch = elements.fixed_epoch
        if min_epoch is not None:
            trainer_config["min_epoch"] = min_epoch
        if num_epoch is not None:
            trainer_config["num_epoch"] = num_epoch
        if max_epoch is not None:
            trainer_config["max_epoch"] = max_epoch
        if elements.max_snapshot_num is not None:
            trainer_config["max_snapshot_num"] = elements.max_snapshot_num
        if elements.clip_norm is not None:
            trainer_config["clip_norm"] = elements.clip_norm
        # model general
        model_config = elements.model_config or {}
        if elements.ema_decay is not None:
            model_config["ema_decay"] = elements.ema_decay
        kwargs["model_config"] = model_config
        # metrics
        metric_config_: Dict[str, Any] = {}
        if elements.metric_config is not None:
            metric_config_ = elements.metric_config
        elif elements.metrics is not None:
            metric_config_["types"] = elements.metrics
        if metric_config_:
            trainer_config["metric_config"] = metric_config_
        # optimizers
        optimizer = elements.optimizer
        scheduler = elements.scheduler
        optimizers = elements.optimizers
        optimizer_config = elements.optimizer_config
        scheduler_config = elements.scheduler_config
        if optimizers is not None:
            if optimizer is not None:
                print(
                    f"{LoggingMixin.warning_prefix}`optimizer` is set to "
                    f"'{optimizer}' but `optimizers` is provided, so "
                    "`optimizer` will be ignored"
                )
            if optimizer_config is not None:
                print(
                    f"{LoggingMixin.warning_prefix}`optimizer_config` is "
                    f"set to '{optimizer_config}' but `optimizers` is provided, "
                    "so `optimizer_config` will be ignored"
                )
        else:
            preset_optimizer = {}
            if optimizer is not None:
                if optimizer_config is None:
                    optimizer_config = {}
                preset_optimizer = {
                    "optimizer": optimizer,
                    "optimizer_config": optimizer_config,
                }
            if scheduler is not None:
                if scheduler_config is None:
                    scheduler_config = {}
                preset_optimizer.update(
                    {"scheduler": scheduler, "scheduler_config": scheduler_config}
                )
            if preset_optimizer:
                optimizers = {"all": preset_optimizer}
        if optimizers is not None:
            trainer_config["optimizers"] = optimizers
        # misc
        kwargs.update({
            "cuda": elements.cuda,
            "trial": elements.trial,
            "tracker_config": elements.tracker_config,
        })
        if elements.verbose_level is not None:
            kwargs["verbose_level"] = elements.verbose_level
        return cls.from_config(kwargs)


__all__ = ["configs_dict", "Configs", "Elements", "Environment"]
