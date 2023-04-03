import os
import torch

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from dataclasses import dataclass
from cftool.dist import Parallel
from cftool.misc import shallow_copy_dict
from cftool.misc import Serializer
from cftool.misc import ISerializable
from cftool.misc import DataClassBase
from cftool.misc import PureFromInfoMixin

from .task import Task
from ...data import MLData
from ...schema import DLConfig
from ...pipeline import DLEvaluationPipeline


def _task(
    task: Task,
    execute: str,
    workspace: str,
    cuda: Optional[Union[int, str]] = None,
) -> None:
    task_folder = os.path.join(workspace, "__task__")
    task.run(execute, task_folder, cuda)


def inject_distributed_tqdm_kwargs(
    i: int,
    num_jobs: int,
    config: DLConfig,
) -> None:
    tqdm_settings = config.tqdm_settings or {}
    tqdm_settings.setdefault("use_tqdm", True)
    tqdm_settings.setdefault("use_step_tqdm", False)
    tqdm_settings.setdefault("in_distributed", True)
    tqdm_settings.setdefault("tqdm_position", i % (num_jobs or 1) + 1)
    tqdm_settings.setdefault("tqdm_desc", f"epoch (task {i})")
    config.tqdm_settings = tqdm_settings


delimiter = "$^_^$"


def tuple_key_to_str_key(d: Dict[Tuple[str, str], Any]) -> Dict[str, Any]:
    return {delimiter.join(k): v for k, v in d.items()}


def str_key_to_tuple_key(d: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    return {tuple(k.split(delimiter)): v for k, v in d.items()}  # type: ignore


@dataclass
class ExperimentResults(DataClassBase):
    workspaces: List[str]
    workspace_keys: List[Tuple[str, str]]
    pipelines: Optional[List[DLEvaluationPipeline]]


class Experiment(PureFromInfoMixin, ISerializable):
    d: Dict[str, Type["Experiment"]] = {}
    tasks_folder = "__tasks__"
    default_root_workspace = "_experiment"

    def __init__(
        self,
        *,
        num_jobs: int = 1,
        use_cuda: bool = True,
        available_cuda_list: Optional[List[int]] = None,
        resource_config: Optional[Dict[str, Any]] = None,
    ):
        use_cuda = use_cuda and torch.cuda.is_available()
        if available_cuda_list is None and not use_cuda:
            available_cuda_list = []
        self.num_jobs = num_jobs
        self.use_cuda = use_cuda
        self.cuda_list = available_cuda_list
        self.resource_config = resource_config or {}
        self.tasks: Dict[Tuple[str, str], Task] = {}
        self.key_indices: Dict[Tuple[str, str], int] = {}
        self.executes: Dict[Tuple[str, str], str] = {}
        self.workspaces: Dict[Tuple[str, str], str] = {}
        self.results: Optional[ExperimentResults] = None

    @staticmethod
    def data_folder(workspace: Optional[str] = None) -> str:
        workspace = workspace or Experiment.default_root_workspace
        return os.path.join(workspace, "__data__")

    @staticmethod
    def dump_data(
        data: MLData,
        *,
        workspace: Optional[str] = None,
        data_folder: Optional[str] = None,
    ) -> str:
        if data_folder is None:
            data_folder = Experiment.data_folder(workspace)
        os.makedirs(data_folder, exist_ok=True)
        Serializer.save(data_folder, data)
        return data_folder

    @staticmethod
    def fetch_data(
        *,
        workspace: Optional[str] = None,
        data_folder: Optional[str] = None,
    ) -> MLData:
        if data_folder is None:
            data_folder = Experiment.data_folder(workspace)
        return Serializer.load(data_folder, MLData)

    @staticmethod
    def workspace(workspace_key: Tuple[str, str], root_workspace: str) -> str:
        workspace = os.path.join(root_workspace, *workspace_key)
        return os.path.abspath(workspace)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    def add_task(
        self,
        data: Optional[MLData] = None,
        *,
        model: str = "fcnn",
        execute: str = "basic",
        root_workspace: Optional[str] = None,
        workspace_key: Optional[Tuple[str, str]] = None,
        config: Optional[DLConfig] = None,
        data_folder: Optional[str] = None,
        run_command: Optional[str] = None,
        task_meta_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if workspace_key is None:
            counter = 0
            while True:
                workspace_key = model, str(counter)
                if workspace_key not in self.tasks:
                    break
                counter += 1
        if workspace_key in self.tasks:
            raise ValueError(f"task already exists with '{workspace_key}'")
        root_workspace = root_workspace or self.default_root_workspace
        workspace = self.workspace(workspace_key, root_workspace)
        new_idx = len(self.tasks)
        if config is not None:
            config: DLConfig = config.copy()  # type: ignore
            config.workspace = workspace
            config.create_sub_workspace = False
            config.model_name = model
            inject_distributed_tqdm_kwargs(new_idx, self.num_jobs, config)
        if data_folder is None and data is not None:
            data_folder = self.dump_data(data, workspace=workspace)
        new_task = Task.init(
            config,
            run_command,
            model=model,
            workspace=workspace,
            data_folder=None if data_folder is None else os.path.abspath(data_folder),
            **shallow_copy_dict(task_meta_kwargs or {}),
        )
        self.tasks[workspace_key] = new_task
        self.key_indices[workspace_key] = new_idx
        self.executes[workspace_key] = execute
        self.workspaces[workspace_key] = workspace
        return workspace

    def run_tasks(
        self,
        *,
        use_tqdm: bool = True,
        task_loader: Optional[Callable[[str], DLEvaluationPipeline]] = None,
        **parallel_kwargs: Any,
    ) -> ExperimentResults:
        resource_config = shallow_copy_dict(self.resource_config)
        gpu_config = resource_config.setdefault("gpu_config", {})
        gpu_config["available_cuda_list"] = self.cuda_list
        parallel = Parallel(
            self.num_jobs,
            use_tqdm=use_tqdm,
            use_cuda=self.use_cuda,
            resource_config=resource_config,
            **parallel_kwargs,
        )
        sorted_workspace_keys = sorted(self.tasks, key=self.key_indices.get)  # type: ignore
        sorted_tasks = [self.tasks[key] for key in sorted_workspace_keys]
        sorted_executes = [self.executes[key] for key in sorted_workspace_keys]
        sorted_workspaces = [self.workspaces[key] for key in sorted_workspace_keys]
        parallel(_task, sorted_tasks, sorted_executes, sorted_workspaces)
        if task_loader is None:
            pipelines = None
        else:
            pipelines = list(map(task_loader, sorted_workspaces))
        self.results = ExperimentResults(
            sorted_workspaces,
            sorted_workspace_keys,
            pipelines,
        )
        return self.results

    def to_info(self) -> Dict[str, Any]:
        tasks = {k: v.to_pack().asdict() for k, v in self.tasks.items()}
        return dict(
            num_jobs=self.num_jobs,
            use_cuda=self.use_cuda,
            cuda_list=self.cuda_list,
            resource_config=self.resource_config,
            tasks=tuple_key_to_str_key(tasks),
            key_indices=tuple_key_to_str_key(self.key_indices),
            executes=tuple_key_to_str_key(self.executes),
            workspaces=tuple_key_to_str_key(self.workspaces),
            results=None if self.results is None else self.results.asdict(),
        )

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        self.tasks = str_key_to_tuple_key(self.tasks)  # type: ignore
        self.tasks = {k: Task.from_pack(v) for k, v in self.tasks.items()}
        self.key_indices = str_key_to_tuple_key(self.key_indices)  # type: ignore
        self.executes = str_key_to_tuple_key(self.executes)  # type: ignore
        self.workspaces = str_key_to_tuple_key(self.workspaces)  # type: ignore
        if self.results is not None:
            workspace_keys = list(map(tuple, self.results["workspace_keys"]))
            self.results["workspace_keys"] = workspace_keys
            self.results = ExperimentResults(**self.results)


Experiment.register("experiment")(Experiment)


__all__ = [
    "inject_distributed_tqdm_kwargs",
    "Experiment",
    "ExperimentResults",
]
