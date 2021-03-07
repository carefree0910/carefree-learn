import os

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from typing import Any
from typing import Optional
from typing import NamedTuple
from cftool.misc import update_dict
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cflearn.types import data_type
from cflearn.types import general_config_type

from .basic import make
from .basic import save
from .basic import load
from ..configs import _parse_config
from ..pipeline import Pipeline


def _cleanup() -> None:
    dist.destroy_process_group()


def _fit(
    rank: int,
    workplace: str,
    train_file: str,
    valid_file: Optional[str],
    model: str,
    config_file: str,
) -> None:
    print(f"{LoggingMixin.info_prefix}initializing rank {rank}")
    config = _parse_config(config_file)
    config["rank"] = rank
    if rank != 0:
        config["show_summary"] = False
        config["verbose_level"] = 0
        config.setdefault("trigger_logging", False)
    m = make(model, config).fit(train_file, x_cv=valid_file)
    dist.barrier()
    if rank == 0:
        save(m, saving_folder=workplace)
    _cleanup()


class DDPResults(NamedTuple):
    m: Pipeline
    train_file: str
    valid_file: Optional[str]
    config_file: str


def ddp(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    workplace: str = "__ddp__",
    config: general_config_type = None,
    increment_config: general_config_type = None,
    world_size: int = 1,
    **kwargs: Any,
) -> DDPResults:
    os.makedirs(workplace, exist_ok=True)
    # logging
    default_logging_folder = os.path.join(workplace, "_logs")
    kwargs.setdefault("logging_folder", default_logging_folder)
    # data
    data_folder = os.path.join(workplace, "__data__")
    os.makedirs(data_folder, exist_ok=True)
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
    final_config["world_size"] = world_size
    # save configs
    config_file = Saving.save_dict(final_config, "config", workplace)
    # spawn
    mp.spawn(
        _fit,
        args=(workplace, train_file, valid_file, model, config_file),
        nprocs=world_size,
        join=True,
    )
    # gather
    m = load(saving_folder=workplace)[model][0]
    return DDPResults(m, train_file, valid_file, config_file)


__all__ = [
    "ddp",
]
