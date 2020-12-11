import os
import json
import argparse

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import NamedTuple
from cftool.misc import Saving
from cfdata.tabular import data_type

meta_config_name = "__meta__"
data_config_file = "__data__.json"


class Info(NamedTuple):
    workplace: str
    meta: Dict[str, Any]
    kwargs: Dict[str, Any]
    data_list: Optional[List[data_type]]


def get_info(*, requires_data: bool = True) -> Info:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str)
    args = parser.parse_args()
    # common
    meta_config = Saving.load_dict(meta_config_name, args.config_folder)
    cuda = meta_config["cuda"]
    kwargs = meta_config["config"]
    workplace = meta_config["workplace"]
    kwargs["cuda"] = cuda
    kwargs["logging_folder"] = os.path.join(workplace, "_logs")
    kwargs.setdefault("log_pipeline_to_artifacts", True)
    # data
    if not requires_data:
        data_list = None
    else:
        data_folder = kwargs.get("data_folder")
        if data_folder is None:
            raise ValueError("`data_folder` should be provided")
        data_config_path = os.path.join(data_folder, data_config_file)
        keys = ["x", "y", "x_cv", "y_cv"]
        if os.path.isfile(data_config_path):
            with open(data_config_path, "r") as f:
                data_config = json.load(f)
            data_list = list(map(data_config.get, keys))
        else:
            data_list = []
            for key in keys:
                data_file = os.path.join(data_folder, f"{key}.npy")
                if not os.path.isfile(data_file):
                    data_list.append(None)
                else:
                    data_list.append(np.load(data_file))
    return Info(workplace, meta_config, kwargs, data_list)


__all__ = [
    "get_info",
]
