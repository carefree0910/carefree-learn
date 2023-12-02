import numpy as np

from enum import Enum
from typing import Optional
from typing import NamedTuple
from cftool.types import np_dict_type


class MLDatasetTag(str, Enum):
    TRAIN = "train"
    VALID = "validation"


class MLBatch(NamedTuple):
    input: np.ndarray
    labels: Optional[np.ndarray]
    others: Optional[np_dict_type] = None


__all__ = [
    "MLDatasetTag",
    "MLBatch",
]
