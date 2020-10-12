import logging

from typing import *
from cftool.misc import *
from abc import ABCMeta, abstractmethod

from ..bases import *
from .basic import make


zoo_dict: Dict[str, Type["ZooBase"]] = {}


class ZooBase(LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        model_type: str = "default",
        increment_config: Dict[str, Any] = None,
    ):
        self._model_type = model_type
        self._increment_config = increment_config

    @property
    @abstractmethod
    def benchmarks(self) -> Dict[str, dict]:
        """
        this method should return a dict of configs (which represent benchmarks)
        * Note that "default" key should always be included in the returned dict
        """
        raise NotImplementedError

    @property
    def config(self) -> dict:
        """ return corresponding config of self._model_type, update with increment_config if provided """
        config_dict = self.benchmarks
        assert "default" in config_dict, "'default' should be included in config_dict"
        config = config_dict.get(self._model_type)
        if config is None:
            if self._model_type != "default":
                self.log_msg(
                    f"model_type '{self._model_type}' is not recognized, 'default' model_type will be used",
                    self.warning_prefix,
                    2,
                    msg_level=logging.WARNING,
                )
                self._model_type = "default"
            config = self.benchmarks["default"]
        config = shallow_copy_dict(config)
        if self._increment_config is not None:
            update_dict(self._increment_config, config)
        return config

    @property
    def model(self) -> str:
        return self._model_type

    @property
    def m(self) -> Wrapper:
        """ return corresponding model of self.config """
        return make(self.model, **self.config)

    def switch(self, model_type) -> "ZooBase":
        """ switch to another model_type """
        self._model_type = model_type
        return self

    @classmethod
    def register(cls, name: str):
        global zoo_dict

        def before(cls_):
            cls_.__identifier__ = name

        return register_core(name, zoo_dict, before_register=before)


@ZooBase.register("fcnn")
class FCNNZoo(ZooBase):
    @property
    def benchmarks(self) -> Dict[str, dict]:
        return {
            "default": {},
            "light_bn": {"model_config": {"hidden_units": [128]}},
            "on_large": {
                "model_config": {
                    "mapping_configs": {"dropout": 0.1, "batch_norm": False}
                }
            },
            "light": {
                "model_config": {
                    "hidden_units": [128],
                    "mapping_configs": {"batch_norm": False},
                }
            },
            "on_sparse": {
                "optimizer_config": {"lr": 1e-4},
                "model_config": {
                    "hidden_units": [128],
                    "mapping_configs": {"dropout": 0.9, "batch_norm": False},
                },
            },
        }


@ZooBase.register("tree_dnn")
class TreeDNNZoo(ZooBase):
    @property
    def benchmarks(self) -> Dict[str, dict]:
        return {
            "default": {},
            "on_large": {
                "model_config": {
                    "dndf_config": None,
                    "mapping_configs": {"dropout": 0.1},
                }
            },
            "light": {
                "model_config": {
                    "dndf_config": None,
                    "mapping_configs": {"batch_norm": False},
                    "default_encoding_configs": {"embedding_dim": 8},
                }
            },
            "on_sparse": {
                "optimizer_config": {"lr": 1e-4},
                "model_config": {
                    "dndf_config": None,
                    "mapping_configs": {
                        "dropout": 0.9,
                        "batch_norm": False,
                        "pruner_config": None,
                    },
                    "default_encoding_configs": {"embedding_dim": 8},
                },
            },
        }


@ZooBase.register("ddr")
class DDRZoo(ZooBase):
    @property
    def benchmarks(self) -> Dict[str, dict]:
        return {
            "default": {},
            "disjoint": {"joint_training": False},
            "q_only": {"fetches": ["quantile"]},
        }


def zoo(
    model: str = "fcnn",
    *,
    model_type: str = "default",
    increment_config: Dict[str, Any] = None,
) -> ZooBase:
    return zoo_dict[model](model_type=model_type, increment_config=increment_config)


__all__ = [
    "ZooBase",
    "zoo",
]
