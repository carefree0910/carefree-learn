import os
import json
import mlflow
import shutil
import getpass
import platform

from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import lock_manager
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.tracking.fluent import _RUN_ID_ENV_VAR

from ...trainer import Trainer
from ...trainer import TrainerCallback
from ...protocol import TrainerState
from ...protocol import MetricsOutputs
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import WARNING_PREFIX


def parse_mlflow_uri(path: str) -> str:
    delim = "/" if platform.system() == "Windows" else ""
    return f"file://{delim}{path}"


@TrainerCallback.register("_inject_loader_name")
class _InjectLoaderName(TrainerCallback):
    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "Trainer",
    ) -> None:
        kwargs["loader_name"] = trainer.train_loader.name  # type: ignore


@TrainerCallback.register("mlflow")
class MLFlowCallback(TrainerCallback):
    def __init__(
        self,
        experiment_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        run_tags: Optional[Dict[str, Any]] = None,
        tracking_folder: str = os.getcwd(),
    ):
        super().__init__()
        tracking_folder = os.path.abspath(tracking_folder)
        tracking_dir = os.path.join(tracking_folder, "mlruns")
        with lock_manager(tracking_folder, ["mlruns"]):
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_uri = parse_mlflow_uri(tracking_dir)
            self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)

        run = None
        from_external = False
        if _RUN_ID_ENV_VAR in os.environ:
            existing_run_id = os.environ[_RUN_ID_ENV_VAR]
            del os.environ[_RUN_ID_ENV_VAR]
            try:
                run = self.mlflow_client.get_run(existing_run_id)
                from_external = True
            except MlflowException:
                print(
                    f"{WARNING_PREFIX}`run_id` is found in environment but "
                    "corresponding mlflow run does not exist. This might cause by "
                    "external calls."
                )

        if run is None:
            if run_tags is None:
                run_tags = {}
            run_tags.setdefault(MLFLOW_USER, getpass.getuser())
            if run_name is not None:
                if run_name_prefix is not None:
                    run_name = f"{run_name_prefix}_{run_name}"
                run_tags.setdefault(MLFLOW_RUN_NAME, run_name)
            run = self.mlflow_client.create_run(experiment_id, tags=run_tags)
        self.run_id = run.info.run_id

        if not from_external:
            for key, value in (params or {}).items():
                self.mlflow_client.log_param(self.run_id, key, value)

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        self.mlflow_client.log_metric(self.run_id, key, lr, step=state.step)

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        for key, value in metric_outputs.metric_values.items():
            self.mlflow_client.log_metric(self.run_id, key, value, step=state.step)

    def log_artifacts(self, trainer: Trainer) -> None:
        self.mlflow_client.log_artifacts(self.run_id, trainer.workplace)

    def finalize(self, trainer: Trainer) -> None:
        self.log_artifacts(trainer)
        self.mlflow_client.set_terminated(self.run_id)


class ArtifactCallback(TrainerCallback):
    key: str

    def __init__(self, num_keep: int = 25):
        super().__init__()
        self.num_keep = num_keep

    def _prepare_folder(self, trainer: Trainer) -> str:
        state = trainer.state
        sub_folder = os.path.join(trainer.workplace, self.key)
        os.makedirs(sub_folder, exist_ok=True)
        current_steps = sorted(os.listdir(sub_folder))
        if len(current_steps) >= self.num_keep:
            must_keep = set()
            checkpoint_folder = trainer.checkpoint_folder
            if checkpoint_folder is not None:
                score_path = os.path.join(checkpoint_folder, SCORES_FILE)
                with open(score_path, "r") as f:
                    for key in json.load(f):
                        name = os.path.splitext(key)[0]
                        must_keep.add(name[len(PT_PREFIX) :])
            num_left = len(current_steps)
            for step in current_steps:
                if step in must_keep:
                    continue
                shutil.rmtree(os.path.join(sub_folder, step))
                num_left -= 1
                if num_left < self.num_keep:
                    break
        sub_folder = os.path.join(sub_folder, str(state.step))
        os.makedirs(sub_folder, exist_ok=True)
        return sub_folder


__all__ = [
    "MLFlowCallback",
    "ArtifactCallback",
]
