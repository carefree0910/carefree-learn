import os
import json
import math

import numpy as np

from typing import Any
from typing import Dict
from typing import NamedTuple
from argparse import Namespace
from cftool.misc import update_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cflearn.types import data_type
from cflearn.types import general_config_type

from ..configs import _parse_config
from ..pipeline import Pipeline

cli_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cli")
cli_root = os.path.abspath(cli_root)


class DeepspeedResults(NamedTuple):
    m: Pipeline
    workplace: str
    config_path: str
    ds_config_path: str


def impute_deepspeed_args(
    args: Namespace,
    config: Dict[str, Any],
    trainer_config: Dict[str, Any],
) -> None:
    if args.deepspeed_config is not None:
        config["use_tqdm"] = False
        config["trigger_logging"] = True
        config["ds_args"] = args
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        ds_config_changed = False
        if trainer_config["use_amp"]:
            ds_config_changed = True
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        clip_norm = trainer_config["clip_norm"]
        if clip_norm > 0.0:
            ds_config_changed = True
            ds_config["gradient_clipping"] = clip_norm
        if ds_config_changed:
            folder, file = os.path.split(args.deepspeed_config)
            name, ext = os.path.splitext(file)
            new_file = f"_cf_{name}{ext}"
            new_path = new_file if not folder else os.path.join(folder, new_file)
            with lock_manager(folder or "./", new_file):
                with open(new_path, "w") as f:
                    json.dump(ds_config, f)
            args.deepspeed_config = new_path


def impute_deepspeed_config(
    cuda: str,
    default_batch_size: int,
    ds_config: general_config_type,
    final_config: Dict[str, Any],
) -> Dict[str, Any]:
    parsed_ds_config = _parse_config(ds_config)
    num_cuda = len(cuda.split(","))
    batch_size = parsed_ds_config.setdefault("train_batch_size", default_batch_size)
    micro_batch_size = 2 ** (math.floor(math.log2(batch_size / num_cuda)))
    parsed_ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    new_batch_size = micro_batch_size * num_cuda
    if batch_size != new_batch_size:
        print(
            f"{LoggingMixin.warning_prefix}`batch_size` "
            f"will be switched from {batch_size} to {new_batch_size}"
        )
        parsed_ds_config["train_batch_size"] = new_batch_size
        batch_size = new_batch_size
    parsed_ds_config.setdefault("steps_per_print", 30000 // batch_size)
    final_config["batch_size"] = batch_size
    return parsed_ds_config


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
    # deepspeed
    parsed_ds_config = impute_deepspeed_config(cuda, 256, ds_config, final_config)
    # save configs
    config_file = Saving.save_dict(final_config, "config", workplace)
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
    "impute_deepspeed_args",
    "deepspeed",
]
