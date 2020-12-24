import logging

from abc import *
from typing import *
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin

from .basic import make
from ..pipeline import Pipeline

registered_benchmarks: Dict[str, Dict[str, Dict[str, Any]]] = {}


class Zoo(LoggingMixin, metaclass=ABCMeta):
    token = "_zoo_"

    def __init__(
        self,
        model: str,
        *,
        model_type: str = "default",
        increment_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.increment_config = increment_config

    @property
    def benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        this method should return a dict of configs (which represent benchmarks)
        * Note that "default" key should always be included in the returned dict
        """
        return registered_benchmarks[self.model]

    @property
    def config(self) -> dict:
        """
        return corresponding config of `self.model_type`
        * and update with `increment_config` if provided
        """
        benchmarks = self.benchmarks
        assert "default" in benchmarks, "'default' should be included in config_dict"
        config = benchmarks.get(self.model_type)
        if config is None:
            if self.model_type != "default":
                self.log_msg(
                    f"model_type '{self.model_type}' is not recognized, "
                    "'default' model_type will be used",
                    self.warning_prefix,
                    2,
                    msg_level=logging.WARNING,
                )
                self.model_type = "default"
            config = self.benchmarks["default"]
        new_config = shallow_copy_dict(config)
        if self.increment_config is not None:
            update_dict(self.increment_config, new_config)
        return new_config

    @property
    def m(self) -> Pipeline:
        """ return corresponding model of self.config """
        return make(self.model, **self.config)

    @classmethod
    def parse_type(cls, model_type: str) -> str:
        return f"{cls.token}{model_type}"

    @classmethod
    def register(
        cls,
        model: str,
        model_type: str,
        *,
        transform_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        extractor_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        head_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        increment_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        global registered_benchmarks
        model_dict = registered_benchmarks.setdefault(model, {})
        pipe_configs: Dict[str, Any] = {}
        if transform_configs is not None:
            for pipe, transform_config in transform_configs.items():
                pipe_configs.setdefault(pipe, {})["transform"] = transform_config
        if extractor_configs is not None:
            for pipe, extractor_config in extractor_configs.items():
                pipe_configs.setdefault(pipe, {})["extractor"] = extractor_config
        if head_configs is not None:
            for pipe, head_config in head_configs.items():
                pipe_configs.setdefault(pipe, {})["head"] = head_config
        config = {}
        if pipe_configs:
            config = {"model_config": {"pipe_configs": pipe_configs}}
        if increment_configs is not None:
            update_dict(increment_configs, config)
        model_dict[model_type] = config


# fcnn

Zoo.register("fcnn", "default")
Zoo.register("fcnn", "light_bn", head_configs={"fcnn": {"hidden_units": [128]}})
Zoo.register(
    "fcnn",
    "on_large",
    head_configs={"fcnn": {"mapping_configs": {"dropout": 0.1, "batch_norm": False}}},
)
Zoo.register(
    "fcnn",
    "light",
    head_configs={
        "fcnn": {"hidden_units": [128], "mapping_configs": {"batch_norm": False}}
    },
)
Zoo.register(
    "fcnn",
    "on_sparse",
    head_configs={
        "fcnn": {
            "hidden_units": [128],
            "mapping_configs": {"dropout": 0.9, "batch_norm": False},
        }
    },
    increment_configs={"optimizer_config": {"lr": 1e-4}},
)

# tree dnn

Zoo.register("tree_dnn", "default")
Zoo.register(
    "tree_dnn",
    "on_large",
    head_configs={
        "dndf": {"dndf_config": None},
        "fcnn": {"mapping_configs": {"dropout": 0.1}},
    },
)
Zoo.register(
    "tree_dnn",
    "light",
    head_configs={
        "dndf": {"dndf_config": None},
        "fcnn": {"mapping_configs": {"batch_norm": False}},
    },
    increment_configs={
        "model_config": {"default_encoding_configs": {"embedding_dim": 8}}
    },
)
Zoo.register(
    "tree_dnn",
    "on_sparse",
    head_configs={
        "dndf": {"dndf_config": None},
        "fcnn": {
            "mapping_configs": {
                "dropout": 0.9,
                "batch_norm": False,
                "pruner_config": None,
            }
        },
    },
    increment_configs={
        "optimizer_config": {"lr": 1e-4},
        "model_config": {"default_encoding_configs": {"embedding_dim": 8}},
    },
)


__all__ = ["Zoo"]
