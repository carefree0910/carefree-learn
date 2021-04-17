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

from cflearn.constants import DATA_CONFIG_FILE
from cflearn.constants import META_CONFIG_NAME


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
    meta_config = Saving.load_dict(META_CONFIG_NAME, args.config_folder)
    kwargs = meta_config["config"]
    workplace = meta_config["workplace"]
    kwargs["workplace"] = workplace
    # data
    if not requires_data:
        data_list = None
    else:
        data_folder = kwargs.pop("data_folder", None)
        if data_folder is None:
            raise ValueError("`data_folder` should be provided")
        data_config_path = os.path.join(data_folder, DATA_CONFIG_FILE)
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
