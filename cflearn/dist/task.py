import os
import sys
import platform

import numpy as np

from typing import *
from cftool.misc import Saving, shallow_copy_dict

from ..types import data_type

IS_LINUX = platform.system() == "Linux"


class Task:
    def __init__(self, idx: int, model: str, identifier: str, temp_folder: str):
        self.idx = idx
        self.model = model
        self.identifier = identifier
        self.temp_folder = temp_folder
        self.config: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
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

    def prepare(
        self,
        x: data_type = None,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        external: bool,
        data_task: "Task" = None,
        trains_config: Dict[str, Any] = None,
        tracker_config: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> "Task":
        kwargs["model"] = self.model
        kwargs["logging_folder"] = self.saving_folder
        if tracker_config is not None:
            kwargs["tracker_config"] = tracker_config
        if trains_config is not None:
            kwargs["trains_config"] = trains_config
        if external:
            kwargs["trigger_logging"] = True
            if data_task is not None:
                kwargs["data_folder"] = data_task.saving_folder
            elif not isinstance(x, np.ndarray):
                kwargs["x"], kwargs["y"] = x, y
                kwargs["x_cv"], kwargs["y_cv"] = x_cv, y_cv
            else:
                self.dump_data(x, y)
                self.dump_data(x_cv, y_cv, "_cv")
        self.config = kwargs
        return self

    # external run (use m.trains())

    def dump_data(self, x: data_type, y: data_type = None, postfix: str = "") -> None:
        for key, value in zip([f"x{postfix}", f"y{postfix}"], [x, y]):
            if value is None:
                continue
            np.save(os.path.join(self.saving_folder, f"{key}.npy"), value)

    def fetch_data(
        self,
        postfix: str = "",
        *,
        from_data_folder: bool = False,
    ) -> Tuple[data_type, data_type]:
        if not from_data_folder:
            data_folder: Optional[str] = self.saving_folder
        else:
            assert self.config is not None
            data_folder = self.config.get("data_folder")
            if data_folder is None:
                raise ValueError("data_folder is not prepared")
        data = []
        data_folder = str(data_folder)
        for key in [f"x{postfix}", f"y{postfix}"]:
            file = os.path.join(data_folder, f"{key}.npy")
            data.append(None if not os.path.isfile(file) else np.load(file))
        return data[0], data[1]

    def dump_config(self, config: Dict[str, Any]) -> "Task":
        Saving.save_dict(config, "config", self.saving_folder)
        return self

    def run_external(self, cuda: Optional[int] = None) -> "Task":
        config = shallow_copy_dict(self.config)
        config["cuda"] = cuda
        self.dump_config(config)
        os.system(f"{self.run_command} --config_folder {self.saving_folder}")
        return self

    # internal fit (use m.fit())

    def fit(
        self,
        make: Callable,
        save: Callable,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        sample_weights: np.ndarray = None,
        prepare: bool = True,
        cuda: int = None,
        **kwargs: Any,
    ) -> "Task":
        if prepare:
            self.prepare(x, y, x_cv, y_cv, external=False, **kwargs)
        assert self.config is not None
        m = make(cuda=cuda, **self.config)
        m.fit(x, y, x_cv, y_cv, sample_weights=sample_weights)
        save(m, saving_folder=self.saving_folder)
        return self

    # save & load

    def save(self, saving_folder: str) -> "Task":
        os.makedirs(saving_folder, exist_ok=True)
        Saving.save_dict(
            {
                "idx": self.idx,
                "model": self.model,
                "identifier": self.identifier,
                "temp_folder": self.temp_folder,
                "config": self.config,
            },
            "kwargs",
            saving_folder,
        )
        return self

    @classmethod
    def load(cls, saving_folder: str) -> "Task":
        kwargs = Saving.load_dict("kwargs", saving_folder)
        config = kwargs.pop("config")
        task = cls(**kwargs)
        task.config = config
        return task

    # special

    @classmethod
    def data_task(cls, i: int, identifier: str, experiments: Any) -> "Task":
        return cls(i, "data", identifier, experiments.temp_folder)


__all__ = ["Task"]
