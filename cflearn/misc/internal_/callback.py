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

from ..toolkit import to_device
from ..toolkit import save_images
from ..toolkit import eval_context
from ...trainer import Trainer
from ...trainer import TrainerCallback
from ...protocol import TrainerState
from ...protocol import MetricsOutputs
from ...constants import INPUT_KEY
from ...constants import LABEL_KEY
from ...constants import PT_PREFIX
from ...constants import SCORES_FILE
from ...constants import WARNING_PREFIX
from ...models.cv.protocol import GeneratorMixin


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


@TrainerCallback.register("generator")
class GeneratorCallback(ArtifactCallback):
    key = "images"
    num_interpolations = 16

    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[INPUT_KEY]
        model = trainer.model
        if not isinstance(model, GeneratorMixin):
            msg = "`GeneratorCallback` is only compatible with `GeneratorMixin`"
            raise ValueError(msg)
        is_conditional = model.is_conditional
        labels = None if not is_conditional else batch[LABEL_KEY]
        image_folder = self._prepare_folder(trainer)
        # original
        save_images(original, os.path.join(image_folder, "original.png"))
        # reconstruct
        if model.can_reconstruct:
            with eval_context(model):
                reconstructed = model.reconstruct(original, labels=labels)
            save_images(reconstructed, os.path.join(image_folder, "reconstructed.png"))
        # sample
        with eval_context(model):
            sampled = model.sample(len(original))
        save_images(sampled, os.path.join(image_folder, "sampled.png"))
        # interpolation
        with eval_context(model):
            interpolations = model.interpolate(self.num_interpolations)
        save_images(interpolations, os.path.join(image_folder, "interpolations.png"))
        # conditional sampling
        if not is_conditional:
            return None
        cond_folder = os.path.join(image_folder, "conditional")
        os.makedirs(cond_folder, exist_ok=True)
        with eval_context(model):
            for i in range(model.num_classes):
                sampled = model.sample(len(original), class_idx=i)
                interpolations = model.interpolate(len(original), class_idx=i)
                save_images(sampled, os.path.join(cond_folder, f"sampled_{i}.png"))
                path = os.path.join(cond_folder, f"interpolations_{i}.png")
                save_images(interpolations, path)


__all__ = [
    "MLFlowCallback",
    "ArtifactCallback",
    "GeneratorCallback",
]
