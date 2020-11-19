import os
import json
import torch

from abc import abstractmethod
from abc import ABC
from typing import Any
from typing import Dict
from typing import Type
from typing import Callable
from typing import Optional
from cftool.misc import timestamp
from cftool.misc import update_dict
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None


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


__all__ = ["configs_dict", "Configs", "Environment"]
