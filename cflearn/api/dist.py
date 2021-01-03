import os

import numpy as np

from typing import Any
from typing import NamedTuple
from cftool.misc import update_dict
from cftool.misc import Saving
from cflearn.types import data_type
from cflearn.types import general_config_type

from ..configs import _parse_config
from ..configs import Elements
from ..pipeline import Pipeline

cli_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cli")
cli_root = os.path.abspath(cli_root)


class DeepspeedResults(NamedTuple):
    m: Pipeline
    workplace: str
    config_path: str
    ds_config_path: str


def deepspeed(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    workplace: str = "__deepspeed__",
    config: general_config_type = None,
    ds_config: general_config_type = None,
    increment_config: general_config_type = None,
    cuda: str = "0",
    **kwargs: Any,
) -> DeepspeedResults:
    os.makedirs(workplace, exist_ok=True)
    # logging
    default_logging_folder = os.path.join(workplace, "_logs")
    kwargs.setdefault("logging_folder", default_logging_folder)
    # data
    data_folder = os.path.join(workplace, "__data__")
    if not isinstance(x, np.ndarray):
        train_file = x
        valid_file = x_cv
    else:
        msg_base = "`{}` should be provided when `{}` is a numpy array"
        if y is None:
            raise ValueError(msg_base.format("y", "x"))
        train_file = os.path.join(data_folder, "train.npy")
        np.save(train_file, np.hstack([x, y]))
        valid_file = None
        if x_cv is not None:
            if y_cv is None:
                raise ValueError(msg_base.format("y_cv", "x_cv"))
            valid_file = os.path.join(data_folder, "valid.npy")
            np.save(valid_file, np.hstack([x_cv, y_cv]))
    # config
    parsed_config = update_dict(_parse_config(config), kwargs)
    parsed_increment_config = _parse_config(increment_config)
    final_config = update_dict(parsed_increment_config, parsed_config)
    final_config["model_saving_folder"] = workplace
    config_file = Saving.save_dict(final_config, "config", workplace)
    # deepspeed
    parsed_ds_config = _parse_config(ds_config)
    parsed_ds_config.setdefault("train_batch_size", 256)
    batch_size = final_config.get("batch_size", Elements().batch_size)
    parsed_ds_config["train_micro_batch_size_per_gpu"] = batch_size
    parsed_ds_config.setdefault("steps_per_print", 30000 // batch_size)
    ds_config_file = Saving.save_dict(parsed_ds_config, "ds_config", workplace)

    os.system(
        f"deepspeed "
        f"--include='localhost:{cuda}' "
        f"{os.path.join(cli_root, 'main.py')} "
        f"--model {model} "
        f"--config {config_file} "
        f"--train_file {train_file} "
        f"{'' if valid_file is None else f'--valid_file {valid_file} '}"
        f"--deepspeed_config {ds_config_file}"
    )
    return DeepspeedResults(
        Pipeline.load(os.path.join(workplace, "pipeline")),
        workplace,
        config_file,
        ds_config_file,
    )


__all__ = [
    "deepspeed",
]
