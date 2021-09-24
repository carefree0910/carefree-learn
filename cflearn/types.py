import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional


arr_type = Union[np.ndarray, torch.Tensor]
data_type = Optional[Union[np.ndarray, str]]
param_type = Union[torch.Tensor, torch.nn.Parameter]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
general_config_type = Optional[Union[str, Dict[str, Any]]]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
losses_type = Union[torch.Tensor, tensor_dict_type]
states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]


__all__ = [
    "arr_type",
    "data_type",
    "param_type",
    "configs_type",
    "general_config_type",
    "np_dict_type",
    "tensor_dict_type",
    "losses_type",
    "states_callback_type",
    "sample_weights_type",
]
