import os
import sys

from typing import Any
from typing import Dict
from typing import Type
from typing import Union
from typing import Optional
from cftool.misc import Serializer
from cftool.misc import ISerializable
from cftool.misc import PureFromInfoMixin

from ...schema import DLConfig


class Task(PureFromInfoMixin, ISerializable):
    d: Dict[str, Type["Task"]] = {}
    config: Optional[Dict[str, Any]]
    run_command: Optional[str]
    meta_kwargs: Dict[str, Any]

    @classmethod
    def init(
        cls,
        config: Optional[DLConfig],
        run_command: Optional[str] = None,
        **meta_kwargs: Any,
    ) -> "Task":
        self = cls()
        if config is None:
            self.config = None
        else:
            self.config = config.to_pack().asdict()
        self.run_command = run_command
        self.meta_kwargs = meta_kwargs
        return self

    def run(
        self,
        execute: str,
        task_folder: str,
        cuda: Optional[Union[int, str]],
    ) -> "Task":
        if self.run_command is not None:
            command = self.run_command
        else:
            command = f"{sys.executable} -m cflearn.dist.ml.runs.{execute}"
        self.meta_kwargs["cuda"] = cuda
        os.makedirs(task_folder, exist_ok=True)
        Serializer.save(task_folder, self)
        os.system(f"{command} --task_folder {task_folder}")
        return self

    def to_info(self) -> Dict[str, Any]:
        return dict(
            config=self.config,
            run_command=self.run_command,
            meta_kwargs=self.meta_kwargs,
        )


Task.register("task")(Task)


__all__ = ["Task"]
