import os
import sys

from typing import *
from cftool.misc import Saving, shallow_copy_dict

from .runs._utils import meta_config_name


class Task:
    def __init__(self, run_command: Optional[str] = None, **meta_kwargs: Any):
        self.run_command = run_command
        self.meta_kwargs = meta_kwargs

    def run(
        self,
        execute: str,
        config_folder: str,
        cuda: Optional[Union[int, str]],
    ) -> "Task":
        python = sys.executable
        if self.run_command is not None:
            command = self.run_command
        else:
            command = f"{python} -m {'.'.join(['cflearn', 'dist', 'runs', execute])}"
        meta_config = shallow_copy_dict(self.meta_kwargs)
        meta_config["cuda"] = cuda
        os.makedirs(config_folder, exist_ok=True)
        Saving.save_dict(meta_config, meta_config_name, config_folder)
        os.system(f"{command} --config_folder {config_folder}")
        return self

    def save(self, saving_folder: str) -> "Task":
        os.makedirs(saving_folder, exist_ok=True)
        meta_config = shallow_copy_dict(self.meta_kwargs)
        meta_config["run_command"] = self.run_command
        Saving.save_dict(meta_config, meta_config_name, saving_folder)
        return self

    @classmethod
    def load(cls, saving_folder: str) -> "Task":
        meta_config = Saving.load_dict(meta_config_name, saving_folder)
        return cls(**meta_config)


__all__ = ["Task"]
