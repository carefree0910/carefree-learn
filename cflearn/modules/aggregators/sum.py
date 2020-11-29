from typing import Any

from .base import AggregatorBase
from ...types import tensor_dict_type


@AggregatorBase.register("sum")
class Sum(AggregatorBase):
    def reduce(self, outputs: tensor_dict_type, **kwargs: Any) -> tensor_dict_type:
        values = list(outputs.values())
        output = None
        for value in values:
            if value is None:
                continue
            if output is None:
                output = value
            else:
                output = output + value
        return {"predictions": output}


__all__ = ["Sum"]
