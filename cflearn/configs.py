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
    data_config: Optional[Dict[str, Any]] = None
    task_type: Optional[task_type_type] = None
    use_simplify_data: bool = False
    ts_config: Optional[TimeSeriesConfig] = None
    aggregation: str = "continuous"
    aggregation_config: Optional[Dict[str, Any]] = None
    ts_label_collator_config: Optional[Dict[str, Any]] = None
    delim: Optional[str] = None
    has_column_names: Optional[bool] = None
    read_config: Optional[Dict[str, Any]] = None
    logging_folder: Optional[str] = None
    logging_file: Optional[str] = None
    use_amp: bool = False
    min_epoch: Optional[int] = None
    num_epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    fixed_epoch: Optional[int] = None
    max_snapshot_file: int = 5
    clip_norm: float = 0.0
    ema_decay: float = 0.0
    model_config: Optional[Dict[str, Any]] = None
    metrics: Union[str, List[str]] = "auto"
    metric_config: Optional[Dict[str, Any]] = None
    optimizer: Optional[str] = "adamw"
    scheduler: Optional[str] = "plateau"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    optimizers: Optional[Dict[str, Any]] = None
    extra_config: Optional[Dict[str, Any]] = None
    user_defined_config: Optional[Dict[str, Any]] = None

    affected_mappings = {
        "use_simplify_data": {"simplify"},
        "ts_config": {"time_series_config"},
        "fixed_epoch": {"min_epoch", "num_epoch", "max_epoch"},
    }

    def to_config(self) -> Dict[str, Any]:
        if self.extra_config is None:
            kwargs = {}
        else:
            kwargs = shallow_copy_dict(self.extra_config)
        # inject fields
        spec = inspect.getfullargspec(type(self)).args[1:-2]
        for key in spec:
            value = getattr(self, key)
            if value is not None or key not in kwargs:
                kwargs[key] = value
        # data
        data_config = self.data_config or {}
        task_type = kwargs.pop("task_type")
        if task_type is not None:
            data_config["task_type"] = task_type
        data_config["simplify"] = kwargs.pop("use_simplify_data")
        data_config["time_series_config"] = kwargs.pop("ts_config")
        sampler_config = kwargs.setdefault("sampler_config", {})
        sampler_config["aggregation"] = kwargs.pop("aggregation")
        sampler_config["aggregation_config"] = kwargs.pop("aggregation_config")
        read_config = self.read_config or {}
        read_config["delim"] = kwargs.pop("delim")
        read_config["has_column_names"] = kwargs.pop("has_column_names")
        kwargs["data_config"] = data_config
        kwargs["read_config"] = read_config
        # logging
        if self.logging_folder is not None:
            if self.logging_file is not None:
                logging_file = self.logging_file
            else:
                logging_file = f"{self.model}_{timestamp()}.log"
            kwargs["logging_file"] = logging_file
        # trainer
        trainer_config = kwargs.setdefault("trainer_config", {})
        trainer_config["use_amp"] = kwargs.pop("use_amp")
        min_epoch = kwargs.pop("min_epoch")
        num_epoch = kwargs.pop("num_epoch")
        max_epoch = kwargs.pop("max_epoch")
        if self.fixed_epoch is not None:
            msg = "`{}` should not be provided when `fixed_epoch` is provided"
            if min_epoch is not None:
                raise ValueError(msg.format("min_epoch"))
            if num_epoch is not None:
                raise ValueError(msg.format("num_epoch"))
            if max_epoch is not None:
                raise ValueError(msg.format("max_epoch"))
            min_epoch = num_epoch = max_epoch = self.fixed_epoch
        no_min, no_num, no_max = min_epoch is None, num_epoch is None, max_epoch is None
        default_min, default_num, default_max = 0, 40, 200
        if no_min and no_num and no_max:
            min_epoch = default_min
            num_epoch = default_num
            max_epoch = default_max
        elif no_min and no_num:
            min_epoch = min(default_min, max_epoch)
            num_epoch = min(default_num, max_epoch)
        elif no_min and no_max:
            min_epoch = min(default_min, num_epoch)
            max_epoch = max(default_max, num_epoch)
        elif no_num and no_max:
            num_epoch = max(default_num, min_epoch)
            max_epoch = max(default_max, min_epoch)
        elif no_min:
            if num_epoch > max_epoch:
                raise ValueError("`num_epoch` should not be greater than `max_epoch`")
            min_epoch = min(default_min, num_epoch)
        elif no_num:
            if min_epoch > max_epoch:
                raise ValueError("`min_epoch` should not be greater than `max_epoch`")
            num_epoch = max(min(default_num, max_epoch), min_epoch)
        elif no_max:
            if min_epoch > num_epoch:
                raise ValueError("`min_epoch` should not be greater than `num_epoch`")
            max_epoch = max(default_max, num_epoch)
        trainer_config["min_epoch"] = min_epoch
        trainer_config["num_epoch"] = num_epoch
        trainer_config["max_epoch"] = max_epoch
        trainer_config["max_snapshot_file"] = kwargs.pop("max_snapshot_file")
        trainer_config["clip_norm"] = kwargs.pop("clip_norm")
        # model
        model_config = self.model_config or {}
        model_config["ema_decay"] = kwargs.pop("ema_decay")
        kwargs["model_config"] = model_config
        # metrics
        metric_config = kwargs.pop("metric_config") or {}
        if self.metric_config is None and self.metrics is not None:
            metric_config["types"] = kwargs.pop("metrics")
        trainer_config["metric_config"] = metric_config
        # optimizers
        optimizer = kwargs.pop("optimizer")
        scheduler = kwargs.pop("scheduler")
        optimizers = kwargs.pop("optimizers")
        optimizer_config = kwargs.pop("optimizer_config") or {}
        scheduler_config = kwargs.pop("scheduler_config") or {}
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
            if scheduler is not None:
                print(
                    f"{LoggingMixin.warning_prefix}`scheduler` is set to "
                    f"'{scheduler}' but `optimizers` is provided, so "
                    "`scheduler` will be ignored"
                )
            if scheduler_config is not None:
                print(
                    f"{LoggingMixin.warning_prefix}`scheduler_config` is "
                    f"set to '{scheduler_config}' but `optimizers` is provided, "
                    "so `scheduler_config` will be ignored"
                )
        else:
            preset_optimizer = {}
            if optimizer is not None:
                optimizer_config.setdefault("lr", 1e-3)
                preset_optimizer = {
                    "optimizer": optimizer,
                    "optimizer_config": optimizer_config,
                }
            if scheduler is not None:
                preset_optimizer.update(
                    {
                        "scheduler": scheduler,
                        "scheduler_config": scheduler_config,
                    }
                )
            if preset_optimizer:
                optimizers = {"all": preset_optimizer}
        if optimizers is not None:
            trainer_config["optimizers"] = optimizers
        return kwargs

    @classmethod
    def make(
        cls,
        config: Optional[Dict[str, Any]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
    ) -> "Elements":
        cfg, inc_cfg = map(_parse_config, [config, increment_config])
        kwargs = update_dict(inc_cfg, cfg)
        user_defined_config = shallow_copy_dict(kwargs)
        # pipeline general
        model = kwargs.setdefault("model", "fcnn")
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
        kwargs.setdefault("ts_label_collator_config", {})
        log_folder = kwargs.setdefault("logging_folder", os.path.join("_logs", model))
        log_file = kwargs.get("logging_file")
        if log_file is not None:
            log_path = os.path.join(log_folder, log_file)
        else:
            log_path = os.path.abspath(os.path.join(log_folder, f"{timestamp()}.log"))
        kwargs["_logging_path_"] = log_path
        # trainer general
        trainer_config = kwargs.setdefault("trainer_config", {})
        trainer_config.setdefault("update_binary_threshold_at_runtime", False)
        use_amp = trainer_config.get("use_amp", False)
        trainer_config["use_amp"] = use_amp and amp is not None
        default_checkpoint_folder = os.path.join(log_folder, "checkpoints")
        trainer_config.setdefault("checkpoint_folder", default_checkpoint_folder)
        # trainer -> metrics
        metric_config = trainer_config.setdefault("metric_config", {})
        metric_config.setdefault("decay", 0.1)
        # model general
        model_config = kwargs.setdefault("model_config", {})
        encoding_methods = model_config.get("encoding_methods", {})
        encoding_configs = model_config.get("encoding_configs", {})
        encoding_methods = {str(k): v for k, v in encoding_methods.items()}
        encoding_configs = {str(k): v for k, v in encoding_configs.items()}
        model_config["encoding_methods"] = encoding_methods
        model_config["encoding_configs"] = encoding_configs
        model_config.setdefault("default_encoding_configs", {})
        model_config.setdefault("loss_config", {})
        # misc
        kwargs.setdefault("cuda", None)
        kwargs.setdefault("trial", None)
        kwargs.setdefault("use_tqdm", True)
        kwargs.setdefault("verbose_level", 2)
        kwargs.setdefault("tracker_config", None)
        kwargs.setdefault("trigger_logging", False)
        kwargs.setdefault("use_timing_context", True)
        # convert to `Elements`
        spec = inspect.getfullargspec(cls).args[1:-2]
        main_configs = {key: kwargs.pop(key) for key in spec if key in kwargs}
        existing_extra_config = main_configs.get("extra_config")
        if existing_extra_config is None:
            main_configs["extra_config"] = kwargs
        else:
            update_dict(kwargs, existing_extra_config)
        main_configs["user_defined_config"] = user_defined_config
        return cls(**main_configs)


class Environment:
    def __init__(
        self,
        config: Dict[str, Any],
        user_defined_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        if user_defined_config is None:
            user_defined_config = config
        self.user_defined_config = user_defined_config

    def __getattr__(self, item: str) -> Any:
        return self.config[item]

    def update_default_config(self, new_default_config: Dict[str, Any]) -> None:
        user_affected = set()
        for key in self.user_defined_config.keys():
            user_affected.update(Elements.affected_mappings.get(key, {key}))

        def _core(current: Dict[str, Any], new_default: Dict[str, Any]) -> None:
            for k, new_default_v in new_default.items():
                current_v = current.get(k)
                if current_v is None:
                    current[k] = new_default_v
                elif not isinstance(new_default_v, dict):
                    if k not in user_affected:
                        current[k] = new_default_v
                else:
                    _core(current_v, new_default_v)

        _core(self.config, new_default_config)

    @property
    def device(self) -> torch.device:
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

    @classmethod
    def from_elements(cls, elements: Elements) -> "Environment":
        return cls(elements.to_config(), elements.user_defined_config)


__all__ = ["configs_dict", "Configs", "Elements", "Environment"]
