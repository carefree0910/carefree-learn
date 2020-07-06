import os
import json
import torch
import platform
import subprocess

import numpy as np

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
    def cuda_run_command(self) -> str:
        python = subprocess.check_output(["which", "python"]).decode().strip()
        return f"{python} -m {'.'.join(['cflearn', 'dist', 'cuda_run'])}"

    def fit(self,
            make: callable,
            save: callable,
            x: data_type,
            y: data_type = None,
            x_cv: data_type = None,
            y_cv: data_type = None,
            cuda: int = None,
            **kwargs) -> "Task":
        kwargs["cuda"] = cuda
        kwargs["logging_folder"] = self.saving_folder
        if not IS_LINUX:
            m = make(self.model, **kwargs)
            m.fit(x, y, x_cv, y_cv)
            save(m, saving_folder=self.saving_folder)
        else:
            kwargs["model"] = self.model
            kwargs["trigger_logging"] = True
            config_file = os.path.join(self.saving_folder, "config.json")
            if not isinstance(x, np.ndarray):
                kwargs["x"], kwargs["y"] = x, y
                kwargs["x_cv"], kwargs["y_cv"] = x_cv, y_cv
            else:
                for key, value in zip(["x", "y", "x_cv", "y_cv"], [x, y, x_cv, y_cv]):
                    if value is None:
                        continue
                    np.save(key, value)
            with open(config_file, "w") as f:
                json.dump(kwargs, f)
            os.system(f"{self.cuda_run_command} --config_file {config_file}")
        return self


__all__ = ["Task"]
