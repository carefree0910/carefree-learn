import argparse

from typing import Any
from typing import Dict
from typing import Optional
from typing import NamedTuple
from cftool.misc import Serializer

from cflearn.data import MLData
from cflearn.schema import DLConfig
from cflearn.dist.ml.task import Task


class Info(NamedTuple):
    workplace: str
    meta: Dict[str, Any]
    config: Optional[DLConfig]
    data: Optional[MLData]


def get_info(*, requires_data: bool = True) -> Info:
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_folder", type=str)
    args = parser.parse_args()
    task_folder = args.task_folder
    # common
    task = Serializer.load(task_folder, Task)
    meta_config = task.meta_kwargs
    workspace = meta_config["workspace"]
    if task.config is None:
        config = None
    else:
        config = DLConfig.from_pack(task.config)
    # data
    if not requires_data:
        data = None
    else:
        data_folder = meta_config.pop("data_folder", None)
        if data_folder is None:
            msg = "`data_folder` should be provided when `requires_data` is True"
            raise ValueError(msg)
        data = Serializer.load(data_folder, MLData)
    return Info(workspace, meta_config, config, data)


__all__ = [
    "get_info",
]
