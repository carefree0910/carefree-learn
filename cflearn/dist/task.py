import os
import sys
import json
import platform

import numpy as np

from typing import Any, Dict, Tuple
from cftool.misc import shallow_copy_dict

from ..misc.toolkit import data_type

IS_LINUX = platform.system() == "Linux"


class Task:
    def __init__(self,
                 idx: int,
                 model: str,
                 identifier: str,
                 temp_folder: str):
        self.idx = idx
        self.model = model
        self.identifier = identifier
        self.temp_folder = temp_folder
        self.config = self.config_file = None

    def __str__(self):
        return f"Task({self.identifier}_{self.idx})"

    __repr__ = __str__

    @property
    def saving_folder(self) -> str:
        folder = os.path.join(self.temp_folder, self.identifier, str(self.idx))
        folder = os.path.abspath(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def run_command(self) -> str:
        python = sys.executable
        return f"{python} -m {'.'.join(['cflearn', 'dist', 'run'])}"

    def prepare(self,
                x: data_type,
                y: data_type = None,
                x_cv: data_type = None,
                y_cv: data_type = None,
                *,
                tracker_config: Dict[str, Any] = None,
                trains_config: Dict[str, Any] = None,
                external: bool,
                **kwargs) -> "Task":
        kwargs["model"] = self.model
        kwargs["logging_folder"] = self.saving_folder
        if tracker_config is not None:
            kwargs["tracker_config"] = tracker_config
        if trains_config is not None:
            kwargs["trains_config"] = trains_config
        if external:
            kwargs["trigger_logging"] = True
            self.config_file = os.path.join(self.saving_folder, "config.json")
            if not isinstance(x, np.ndarray):
                kwargs["x"], kwargs["y"] = x, y
                kwargs["x_cv"], kwargs["y_cv"] = x_cv, y_cv
            else:
                for key, value in zip(["x", "y", "x_cv", "y_cv"], [x, y, x_cv, y_cv]):
                    if value is None:
                        continue
                    np.save(os.path.join(self.saving_folder, f"{key}.npy"), value)
        self.config = kwargs
        return self

    # external run (use m.trains())

    def fetch_data(self) -> Tuple[data_type, ...]:
        data = []
        for key in ["x", "y", "x_cv", "y_cv"]:
            file = os.path.join(self.saving_folder, f"{key}.npy")
            data.append(None if not os.path.isfile(file) else np.load(file))
        return tuple(data)

    def dump_config(self,
                    config: Dict[str, Any]) -> "Task":
        with open(self.config_file, "w") as f:
            json.dump(config, f)
        return self

    def run_external(self,
                     cuda: int = None) -> "Task":
        config = shallow_copy_dict(self.config)
        config["cuda"] = cuda
        self.dump_config(config)
        os.system(f"{self.run_command} --config_file {self.config_file}")
        return self

    # internal fit (use m.fit())

    def fit(self,
            make: callable,
            save: callable,
            x: data_type,
            y: data_type = None,
            x_cv: data_type = None,
            y_cv: data_type = None,
            *,
            prepare: bool = True,
            cuda: int = None,
            **kwargs) -> "Task":
        if prepare:
            self.prepare(x, y, x_cv, y_cv, external=False, **kwargs)
        m = make(cuda=cuda, **self.config)
        m.fit(x, y, x_cv, y_cv)
        save(m, saving_folder=self.saving_folder)
        return self


__all__ = ["Task"]
