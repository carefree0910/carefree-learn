import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cfdata.tabular import TaskTypes


data_type = Optional[Union[np.ndarray, List[List[float]], str]]
np_dict_type = Dict[str, Union[np.ndarray, Any]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]
general_config_type = Optional[Union[str, Dict[str, Any]]]
task_type_type = Union[str, TaskTypes]


__all__ = [
    "data_type",
    "np_dict_type",
    "tensor_dict_type",
    "general_config_type",
    "task_type_type",
]
