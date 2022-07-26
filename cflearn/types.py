import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cftool.types import tensor_dict_type


data_type = Optional[Union[np.ndarray, str]]
texts_type = Union[str, List[str]]
param_type = Union[torch.Tensor, torch.nn.Parameter]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
general_config_type = Optional[Union[str, Dict[str, Any]]]
losses_type = Union[torch.Tensor, tensor_dict_type]
states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]


__all__ = [
    "data_type",
    "texts_type",
    "param_type",
    "configs_type",
    "general_config_type",
    "losses_type",
    "states_callback_type",
    "sample_weights_type",
]
