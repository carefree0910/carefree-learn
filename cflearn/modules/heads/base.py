import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from cftool.misc import register_core
from cftool.misc import LoggingMixin

from ..transform import Dimensions
from ...types import tensor_dict_type
from ...configs import configs_dict
from ...configs import Configs
from ...protocol import DataProtocol


head_dict: Dict[str, Type["HeadBase"]] = {}


class HeadConfigs(Configs):
    def __init__(
        self,
        in_dim: int,
        tr_data: DataProtocol,
        tr_weights: Optional[np.ndarray],
        dimensions: Dimensions,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.in_dim = in_dim
        self.tr_data = tr_data
        self.tr_weights = tr_weights
        self.dimensions = dimensions

    @property
    def out_dim(self) -> int:
        out_dim = self.config.get("out_dim")
        if self.tr_data.is_clf:
            default_out_dim = self.tr_data.num_classes
        else:
            default_out_dim = self.tr_data.processed.y.shape[1]
        if out_dim is None:
            out_dim = default_out_dim
        return out_dim

    def inject_dimensions(self, config: Dict[str, Any]) -> None:
        config["in_dim"] = self.in_dim
        config["out_dim"] = self.out_dim

    @classmethod
    def get(
        cls,
        scope: str,
        name: str,
        *,
        in_dim: Optional[int] = None,
        tr_data: Optional[DataProtocol] = None,
        tr_weights: Optional[np.ndarray] = None,
        dimensions: Optional[Dimensions] = None,
        **kwargs: Any,
    ) -> "HeadConfigs":
        if in_dim is None:
            raise ValueError("`in_dim` must be provided for `HeadConfigs`")
        if tr_data is None:
            raise ValueError("`tr_data` must be provided for `HeadConfigs`")
        if dimensions is None:
            raise ValueError("`dimensions` must be provided for `HeadConfigs`")
        cfg_type = configs_dict[scope][name]
        if not issubclass(cfg_type, HeadConfigs):
            raise ValueError(f"'{name}' under '{scope}' scope is not `HeadConfigs`")
        return cfg_type(in_dim, tr_data, tr_weights, dimensions, kwargs)


class HeadBase(nn.Module, LoggingMixin, metaclass=ABCMeta):
    def __init__(self, in_dim: int, out_dim: int, **kwargs: Any):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim

    @abstractmethod
    def forward(self, net: torch.Tensor) -> Union[torch.Tensor, tensor_dict_type]:
        pass

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global head_dict
        return register_core(name, head_dict)

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> "HeadBase":
        return head_dict[name](**config)


__all__ = ["HeadConfigs", "HeadBase"]
