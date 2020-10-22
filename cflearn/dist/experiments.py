import os
import torch
import shutil
import logging

from typing import *
from cftool.dist import Parallel
from cftool.misc import Saving, LoggingMixin, lock_manager

from .task import Task
from ..types import data_type


class Experiments(LoggingMixin):
    def __init__(
        self,
        temp_folder: str = "__tmp__",
        available_cuda_list: Optional[List[int]] = None,
        *,
        overwrite: bool = True,
        use_cuda: bool = True,
    ):
        self.temp_folder = temp_folder
        if os.path.isdir(temp_folder) and overwrite:
            self.log_msg(
                f"'{temp_folder}' already exists, it will be overwritten",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
            shutil.rmtree(temp_folder)
        if available_cuda_list is None and (
            not use_cuda or not torch.cuda.is_available()
        ):
            available_cuda_list = []
        self.cuda_list = available_cuda_list
        self.use_cuda = use_cuda
        self.initialize()

    def initialize(self) -> "Experiments":
        self.tasks: Dict[str, List[Task]] = {}
        self.data_tasks: Dict[str, List[Optional[Task]]] = {}
        return self

    def add_task(
        self,
        x: data_type = None,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        model: str = "fcnn",
        identifier: Optional[str] = None,
        trains_config: Optional[Dict[str, Any]] = None,
        tracker_config: Optional[Dict[str, Any]] = None,
        data_task: Optional[Task] = None,
        **kwargs: Any,
    ) -> "Experiments":
        if identifier is None:
            identifier = model
        if data_task is not None:
            id_data_tasks = self.data_tasks.setdefault(identifier, [])
            if len(id_data_tasks) <= data_task.idx:
                id_data_tasks += [None] * (data_task.idx + 1 - len(id_data_tasks))
            id_data_tasks[data_task.idx] = data_task
        kwargs.setdefault("use_tqdm", False)
        kwargs.setdefault("verbose_level", 0)
        kwargs["trains_config"] = trains_config
        kwargs["tracker_config"] = tracker_config
        current_tasks = self.tasks.setdefault(identifier, [])
        new_task = Task(len(current_tasks), model, identifier, self.temp_folder)
        new_task.prepare(x, y, x_cv, y_cv, external=True, data_task=data_task, **kwargs)
        current_tasks.append(new_task)
        return self

    def run_tasks(
        self,
        *,
        num_jobs: int = 4,
        use_tqdm: bool = True,
        run_tasks: bool = True,
        load_task: Optional[Callable] = None,
    ) -> Dict[str, List[Union[Task, Any]]]:
        def _task(i: int, identifier_: str, cuda: Optional[int] = None) -> Task:
            return self.tasks[identifier_][i].run_external(cuda)

        arguments: List[Tuple[int, str]] = []
        for key in sorted(self.tasks):
            arguments.extend([(i, key) for i in range(len(self.tasks[key]))])

        if not run_tasks:
            tasks = []
            for i, key in arguments:
                tasks.append(self.tasks[key][i])
        else:
            parallel_arguments = list(zip(*arguments))
            if num_jobs <= 1:
                num_jobs = 1
            parallel = Parallel(
                num_jobs,
                use_tqdm=use_tqdm,
                use_cuda=self.use_cuda and torch.cuda.is_available(),
                logging_folder=os.path.join(self.temp_folder, "_parallel_"),
                resource_config={"gpu_config": {"available_cuda_list": self.cuda_list}},
            )
            tasks = parallel(_task, *parallel_arguments).ordered_results

        results: Dict[str, List[Union[Task, Any]]] = {}
        for task, (_, identifier) in zip(tasks, arguments):
            if load_task is None:
                results.setdefault(identifier, []).append(task)
            else:
                loaded = load_task(task)
                results.setdefault(identifier, []).append(loaded)

        return results

    def run(
        self,
        load_task: Optional[Callable],
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        num_jobs: int = 4,
        num_repeat: int = 5,
        models: Union[str, List[str]] = "fcnn",
        identifiers: Optional[Union[str, List[str]]] = None,
        use_tqdm_in_task: bool = False,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> Dict[str, List[Union[Task, Any]]]:
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
                self.add_task(
                    x,
                    y,
                    x_cv,
                    y_cv,
                    model=model,
                    identifier=identifier,
                    **kwargs,
                )

        return self.run_tasks(num_jobs=num_jobs, load_task=load_task, use_tqdm=use_tqdm)

    def save(
        self,
        export_folder: str,
        *,
        simplify: bool = True,
        compress: bool = True,
    ) -> "Experiments":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            # tasks
            tasks_folder = os.path.join(abs_folder, "__tasks__")
            for task_name, tasks in self.tasks.items():
                for i, task in enumerate(tasks):
                    task.save(os.path.join(tasks_folder, task_name, str(i)))
            # data tasks
            mappings = {}
            if self.data_tasks:
                data_tasks_folder = os.path.join(abs_folder, "__data_tasks__")
                for task_name, data_tasks in self.data_tasks.items():
                    local_mappings: List[Optional[str]] = [None] * len(data_tasks)
                    for data_task in data_tasks:
                        if data_task is None:
                            continue
                        idx = data_task.idx
                        tgt_data_folder = os.path.join(
                            data_tasks_folder, data_task.identifier, str(idx)
                        )
                        local_mappings[idx] = tgt_data_folder
                        data_task.save(tgt_data_folder)
                        for file in os.listdir(data_task.saving_folder):
                            if file.endswith(".npy"):
                                shutil.copy(
                                    os.path.join(data_task.saving_folder, file),
                                    os.path.join(tgt_data_folder, file),
                                )
                    mappings[task_name] = local_mappings
            # temp folder
            tgt_temp_folder = os.path.join(abs_folder, "__tmp__")
            if not simplify:
                shutil.copytree(self.temp_folder, tgt_temp_folder)
            else:
                os.makedirs(tgt_temp_folder)
                task_names = set(self.tasks.keys())
                for task_name in os.listdir(self.temp_folder):
                    if task_name not in task_names:
                        continue
                    task_model_folder = os.path.join(self.temp_folder, task_name)
                    for try_idx in os.listdir(task_model_folder):
                        try_idx_folder = os.path.join(task_model_folder, try_idx)
                        tgt_model_folder = os.path.join(
                            tgt_temp_folder, task_name, try_idx
                        )
                        os.makedirs(tgt_model_folder)
                        for file in os.listdir(try_idx_folder):
                            if file == "config.json" or file.endswith(".zip"):
                                shutil.copyfile(
                                    os.path.join(try_idx_folder, file),
                                    os.path.join(tgt_model_folder, file),
                                )
            # kwargs
            kwargs = {
                "available_cuda_list": self.cuda_list,
                "data_tasks_mappings": mappings,
                "use_cuda": self.use_cuda,
            }
            Saving.save_dict(kwargs, "kwargs", abs_folder)
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(cls, saving_folder: str, *, compress: bool = True) -> "Experiments":
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            with Saving.compress_loader(abs_folder, compress, remove_extracted=False):
                tgt_temp_folder = os.path.join(abs_folder, "__tmp__")
                kwargs = Saving.load_dict("kwargs", abs_folder)
                data_tasks_mappings = kwargs.pop("data_tasks_mappings")
                kwargs["overwrite"] = False
                kwargs["temp_folder"] = tgt_temp_folder
                experiments = cls(**kwargs)
                # data tasks
                data_tasks = {}
                data_tasks_folder = os.path.join(abs_folder, "__data_tasks__")
                if os.path.isdir(data_tasks_folder):
                    for task_name, data_tasks_mapping in data_tasks_mappings.items():
                        local_data_tasks: List[Optional[Task]] = [None] * len(
                            data_tasks_mapping
                        )
                        for data_task_folder in data_tasks_mapping:
                            if data_task_folder is None:
                                continue
                            local_data_task = Task.load(data_task_folder)
                            local_data_tasks[local_data_task.idx] = local_data_task
                        data_tasks[task_name] = local_data_tasks
                    experiments.data_tasks = data_tasks
                # tasks
                tasks_folder = os.path.join(abs_folder, "__tasks__")
                experiments.tasks = {}
                for task_name in os.listdir(tasks_folder):
                    local_tasks = experiments.tasks[task_name] = []
                    task_folder = os.path.join(tasks_folder, task_name)
                    corresponding_data_tasks = data_tasks.get(task_name)
                    for try_idx in sorted(map(int, os.listdir(task_folder))):
                        local_task = Task.load(os.path.join(task_folder, str(try_idx)))
                        if corresponding_data_tasks is not None:
                            data_task = corresponding_data_tasks[try_idx]
                            if data_task is not None:
                                assert local_task.config is not None
                                data_folder = data_task.saving_folder
                                local_task.config["data_folder"] = data_folder
                        local_tasks.append(local_task)
        return experiments


__all__ = ["Experiments"]
