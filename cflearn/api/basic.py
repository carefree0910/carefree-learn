import os
import sys

from typing import Any
from typing import List
from typing import Union

from ..types import general_config_type
from ..pipeline import pipeline_dict
from ..pipeline import PipelineProtocol
from ..misc.toolkit import _parse_config


def make(name: str, *, config: general_config_type = None) -> PipelineProtocol:
    config = _parse_config(config)
    return pipeline_dict[name](**config)


def run_ddp(path: str, cuda_list: List[Union[int, str]], **kwargs: Any) -> None:
    def _convert_config() -> str:
        return " ".join([f"--{k}={v}" for k, v in kwargs.items()])

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_list))
    kwargs["nproc_per_node"] = len(cuda_list)
    prefix = f"{sys.executable} -m torch.distributed.run "
    os.system(f"{prefix}{_convert_config()} {path}")


__all__ = [
    "make",
    "run_ddp",
]
