import math
import torch
import logging

import numpy as np
import torch.nn as nn

from typing import *

from cftool.misc import *


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
