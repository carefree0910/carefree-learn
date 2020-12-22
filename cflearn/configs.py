import os
import json
import torch
import inspect

from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from argparse import Namespace
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
    ds_args: Optional[Namespace] = None
    aggregator: str = "sum"
    aggregator_config: Optional[Dict[str, Any]] = None
    production: Optional[str] = None
    data_protocol: str = "tabular"
    loader_protocol: str = "tabular"
    sampler_protocol: str = "tabular"
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
    batch_size: int = 128
    cv_split: Optional[Union[float, int]] = None
    use_amp: bool = False
    min_epoch: Optional[int] = None
    num_epoch: Optional[int] = None
    max_epoch: Optional[int] = None
    fixed_epoch: Optional[int] = None
    max_snapshot_file: int = 5
    clip_norm: float = 0.0
    ema_decay: float = 0.0
    model_config: Optional[Dict[str, Any]] = None
    loss: str = "auto"
    loss_config: Optional[Dict[str, Any]] = None
    default_encoding_init_method: Optional[str] = None
    metrics: Union[str, List[str]] = "auto"
    metric_config: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer: Optional[str] = "adamw"
    scheduler: Optional[str] = "warmup"
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    optimizers: Optional[Dict[str, Any]] = None
    verbose_level: int = 2
    use_tqdm: bool = True
    trigger_logging: bool = False
    log_pipeline_to_artifacts: bool = False
    cuda: Optional[Union[int, str]] = None
    mlflow_config: Optional[Dict[str, Any]] = None
    extra_config: Optional[Dict[str, Any]] = None
    user_config: Optional[Dict[str, Any]] = None
    user_increment_config: Optional[Dict[str, Any]] = None

    @staticmethod
    def affected_mappings() -> Dict[str, Set[str]]:
        return {
            "metrics": {"metric_config*"},
            "use_simplify_data": {"simplify"},
            "ts_config": {"time_series_config*"},
            "fixed_epoch": {"min_epoch", "num_epoch", "max_epoch"},
        }

    @property
    def user_defined_config(self) -> Dict[str, Any]:
        user_config = shallow_copy_dict(self.user_config or {})
        user_increment_config = shallow_copy_dict(self.user_increment_config or {})
        return update_dict(user_increment_config, user_config)

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
        trainer_config.setdefault("use_amp", kwargs.pop("use_amp"))
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
        trainer_config.setdefault("min_epoch", min_epoch)
        trainer_config.setdefault("num_epoch", num_epoch)
        trainer_config.setdefault("max_epoch", max_epoch)
        trainer_config.setdefault("max_snapshot_file", kwargs.pop("max_snapshot_file"))
        trainer_config.setdefault("clip_norm", kwargs.pop("clip_norm"))
        # model
        model_config = self.model_config or {}
        model_config["aggregator"] = kwargs.pop("aggregator")
        model_config["aggregator_config"] = kwargs.pop("aggregator_config") or {}
        model_config["ema_decay"] = kwargs.pop("ema_decay")
        model_config["loss"] = kwargs.pop("loss")
        model_config["loss_config"] = kwargs.pop("loss_config") or {}
        default_encoding_init_method = kwargs.pop("default_encoding_init_method")
        if default_encoding_init_method is not None:
            de_cfg = model_config.setdefault("default_encoding_configs", {})
            de_cfg["init_method"] = default_encoding_init_method
        kwargs["model_config"] = model_config
        # metrics
        metric_config = kwargs.pop("metric_config") or {}
        if self.metric_config is None and self.metrics is not None:
            metric_config["types"] = kwargs.pop("metrics")
        metric_config.setdefault("decay", 0.1)
        trainer_config.setdefault("metric_config", metric_config)
        # optimizers
        lr = kwargs.pop("lr")
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
            if lr is not None:
                optimizer_config["lr"] = lr
            optimizers = {
                "all": {
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "optimizer_config": optimizer_config,
                    "scheduler_config": scheduler_config,
                }
            }
        if optimizers is not None:
            trainer_config.setdefault("optimizers", optimizers)
        # inject user defined configs
        update_dict(self.user_defined_config, kwargs)
        return kwargs

    @classmethod
    def make(
        cls,
        config: Optional[Dict[str, Any]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
    ) -> "Elements":
        cfg, inc_cfg = map(_parse_config, [config, increment_config])
        kwargs = update_dict(shallow_copy_dict(inc_cfg), shallow_copy_dict(cfg))
        user_cfg, user_inc_cfg = map(shallow_copy_dict, [cfg, inc_cfg])
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
        kwargs.setdefault("mlflow_config", None)
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
        main_configs["user_config"] = user_cfg
        main_configs["user_increment_config"] = user_inc_cfg
        return cls(**main_configs)


class Environment:
    def __init__(
        self,
        config: Dict[str, Any],
        user_config: Optional[Dict[str, Any]] = None,
        user_increment_config: Optional[Dict[str, Any]] = None,
        set_device: bool = True,
    ):
        self.config = config
        self.user_config = user_config or {}
        self.user_increment_config = user_increment_config or {}
        # deep speed
        self.is_rank_0 = True
        if self.deepspeed:
            self.is_rank_0 = self.local_rank is None or self.local_rank == 0
            logging_folder = config.pop("logging_folder")
            current_timestamp = timestamp(ensure_different=True)
            config["logging_folder"] = os.path.join(logging_folder, current_timestamp)
            if set_device:
                if self.local_rank is None:
                    self.ds_args.local_rank = 0
                torch.cuda.set_device(self.local_rank)

    def __getattr__(self, item: str) -> Any:
        return self.config[item]

    @property
    def user_defined_config(self) -> Dict[str, Any]:
        user_cfg = shallow_copy_dict(self.user_config)
        user_inc_cfg = shallow_copy_dict(self.user_increment_config)
        return update_dict(user_inc_cfg, user_cfg)

    def update_default_config(self, new_default_config: Dict[str, Any]) -> None:
        user_affected = set()
        affected_mappings = Elements.affected_mappings()

        def _inject_affected(current: Dict[str, Any]) -> None:
            for key, value in current.items():
                user_affected.update(affected_mappings.get(key, {key}))
                if isinstance(value, dict):
                    _inject_affected(value)

        def _core(current: Dict[str, Any], new_default: Dict[str, Any]) -> None:
            for k, new_default_v in new_default.items():
                current_v = current.get(k)
                if current_v is None:
                    current[k] = new_default_v
                else:
                    if f"{k}*" in user_affected:
                        continue
                    if not isinstance(new_default_v, dict):
                        if k not in user_affected:
                            current[k] = new_default_v
                    else:
                        _core(current_v, new_default_v)

        _inject_affected(self.user_defined_config)
        _core(self.config, new_default_config)

    @property
    def deepspeed(self) -> bool:
        return self.ds_args is not None

    @property
    def local_rank(self) -> int:
        return self.ds_args.local_rank

    @property
    def device(self) -> torch.device:
        if self.deepspeed:
            return torch.device("cuda", self.local_rank)
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
    def from_elements(
        cls,
        elements: Elements,
        set_device: bool = True,
    ) -> "Environment":
        return cls(
            elements.to_config(),
            elements.user_config,
            elements.user_increment_config,
            set_device,
        )


__all__ = ["configs_dict", "Configs", "Elements", "Environment"]
