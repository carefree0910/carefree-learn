from ..types import general_config_type
from .internal_.pipeline import pipeline_dict
from .internal_.pipeline import PipelineProtocol
from ..misc.toolkit import _parse_config


def make(name: str, *, config: general_config_type) -> PipelineProtocol:
    config = _parse_config(config)
    return pipeline_dict[name](**config)


__all__ = [
    "make",
]
