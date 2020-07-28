import os
import torch

from typing import *
from cftool.dist import Parallel

from .task import Task
from ..misc.toolkit import data_type


class Experiments:
    def __init__(self,
                 temp_folder: str = "__tmp__",
                 available_cuda_list: List[int] = None):
        self.temp_folder = temp_folder
        if available_cuda_list is None and not torch.cuda.is_available():
            available_cuda_list = []
        self.cuda_list = available_cuda_list
        self.initialize()

    def initialize(self) -> "Experiments":
        self.tasks: Dict[str, List[Task]] = {}
        return self

    def add_task(self,
                 x: data_type,
                 y: data_type = None,
                 x_cv: data_type = None,
                 y_cv: data_type = None,
                 *,
                 model: str = "fcnn",
                 identifier: str = None,
                 trains_config: Dict[str, Any] = None,
                 tracker_config: Dict[str, Any] = None,
                 data_task: Task = None,
                 **kwargs) -> "Experiments":
        if identifier is None:
            identifier = model
        kwargs.setdefault("use_tqdm", False)
        kwargs["trains_config"] = trains_config
        kwargs["tracker_config"] = tracker_config
        current_tasks = self.tasks.setdefault(identifier, [])
        new_task = Task(len(current_tasks), model, identifier, self.temp_folder)
        new_task.prepare(x, y, x_cv, y_cv, external=True, data_task=data_task, **kwargs)
        current_tasks.append(new_task)
        return self

    def run_tasks(self,
                  *,
                  num_jobs: int = 4,
                  load_task: callable = None,
                  use_tqdm: bool = True) -> Dict[str, List[Union[Task, Any]]]:
        def _task(i, identifier_, cuda=None) -> Task:
            return self.tasks[identifier_][i].run_external(cuda)

        arguments = []
        for key in sorted(self.tasks):
            arguments.extend([[i, key] for i in range(len(self.tasks[key]))])
        parallel_arguments = list(zip(*arguments))
        parallel = Parallel(
            num_jobs,
            use_tqdm=use_tqdm,
            use_cuda=torch.cuda.is_available(),
            logging_folder=os.path.join(self.temp_folder, "_parallel_"),
            resource_config={"gpu_config": {"available_cuda_list": self.cuda_list}}
        )
        tasks = parallel(_task, *parallel_arguments).ordered_results
        results = {}
        for task, (_, identifier) in zip(tasks, arguments):
            if load_task is None:
                results.setdefault(identifier, []).append(task)
            else:
                loaded = load_task(task)
                results.setdefault(identifier, []).append(loaded)

        self.initialize()
        return results

    def run(self,
            load_task: callable,
            x: data_type,
            y: data_type = None,
            x_cv: data_type = None,
            y_cv: data_type = None,
            *,
            num_jobs: int = 4,
            num_repeat: int = 5,
            models: Union[str, List[str]] = "fcnn",
            identifiers: Union[str, List[str]] = None,
            use_tqdm_in_task: bool = False,
            use_tqdm: bool = True,
            **kwargs) -> Dict[str, List[Union[Task, Any]]]:
        self.initialize()
        if isinstance(models, str):
            models = [models]
        if identifiers is None:
            identifiers = models.copy()
        elif isinstance(identifiers, str):
            identifiers = [identifiers]
        kwargs["use_tqdm"] = use_tqdm_in_task

        for _ in range(num_repeat):
            for model, identifier in zip(models, identifiers):
                self.add_task(x, y, x_cv, y_cv, model=model, identifier=identifier, **kwargs)

        return self.run_tasks(num_jobs=num_jobs, load_task=load_task, use_tqdm=use_tqdm)


__all__ = ["Experiments"]
