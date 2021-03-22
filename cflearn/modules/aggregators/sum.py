from torch import Tensor
from typing import Any

from .base import AggregatorBase
from ...types import tensor_dict_type


@AggregatorBase.register("sum")
class Sum(AggregatorBase):
    def reduce(self, outputs: tensor_dict_type, **kwargs: Any) -> Tensor:
        return sum(outputs.values())  # type: ignore


__all__ = ["Sum"]
