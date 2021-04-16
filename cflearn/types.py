import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional


data_type = Optional[Union[np.ndarray, str]]
param_type = Union[torch.Tensor, torch.nn.Parameter]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
losses_type = Union[torch.Tensor, tensor_dict_type]


__all__ = [
    "data_type",
    "param_type",
    "np_dict_type",
    "tensor_dict_type",
    "losses_type",
]
