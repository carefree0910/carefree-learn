import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional


data_type = Optional[Union[np.ndarray, List[List[float]], str]]
tensor_dict_type = Dict[str, Union[torch.Tensor, Any]]


__all__ = [
    "data_type",
    "tensor_dict_type",
]
