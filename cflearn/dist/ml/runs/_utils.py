import argparse

from typing import Any
from typing import Dict
from typing import Optional
from typing import NamedTuple
from cftool.misc import Saving

from cflearn.data import MLData
from cflearn.data import DataModule
from cflearn.constants import META_CONFIG_NAME


class Info(NamedTuple):
    workplace: str
    meta: Dict[str, Any]
    kwargs: Dict[str, Any]
    data: Optional[MLData]


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
        data = None
    else:
        data_folder = kwargs.pop("data_folder", None)
        if data_folder is None:
            raise ValueError("`data_folder` should be provided")
        data = DataModule.load(data_folder)
    return Info(workplace, meta_config, kwargs, data)


__all__ = [
    "get_info",
]
