import os
import torch

from typing import *
from itertools import product
from cftool.dist import Parallel

from .task import Task
from ..misc.toolkit import data_type


class Experiments:
    def __init__(self,
                 available_cuda_list: List[int] = None):
        if available_cuda_list is None and not torch.cuda.is_available():
            available_cuda_list = []
        self.cuda_list = available_cuda_list

    def run(self,
            make: callable,
            save: callable,
            load_task: callable,
            x: data_type,
            y: data_type = None,
            x_cv: data_type = None,
            y_cv: data_type = None,
            *,
            num_repeat: int = 5,
            num_parallel: int = 4,
            temp_folder: str = "__tmp__",
            models: Union[str, List[str]] = "fcnn",
            identifiers: Union[str, List[str]] = None,
            return_tasks: bool = False,
            use_tqdm_in_task: bool = False,
            use_tqdm: bool = True,
            **kwargs) -> Dict[str, List[Any]]:
        if isinstance(models, str):
            models = [models]
        if identifiers is None:
            identifiers = models.copy()
        elif isinstance(identifiers, str):
            identifiers = [identifiers]
        kwargs["use_tqdm"] = use_tqdm_in_task

        def _task(i, id_tuple, cuda=None):
            t = Task(i, *id_tuple, temp_folder)
            return t.fit(make, save, x, y, x_cv, y_cv, cuda, **kwargs)

        arguments = list(product(map(str, range(num_repeat)), zip(models, identifiers)))
        parallel_arguments = list(zip(*arguments))
        parallel = Parallel(
            num_parallel,
            use_tqdm=use_tqdm,
            use_cuda=torch.cuda.is_available(),
            logging_folder=os.path.join(temp_folder, "_parallel_"),
            resource_config={"gpu_config": {"available_cuda_list": self.cuda_list}}
        )
        tasks = parallel(_task, *parallel_arguments).ordered_results
        results = {}
        for task, (_, (_, identifier)) in zip(tasks, arguments):
            if return_tasks:
                results.setdefault(identifier, []).append(task)
            else:
                loaded = load_task(task)
                results.setdefault(identifier, []).append(loaded)
        return results


__all__ = ["Experiments"]
