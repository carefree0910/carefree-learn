import os
import time
import shutil

from typing import *
from tqdm.autonotebook import tqdm
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin
from cftool.ml.utils import pattern_type
from cftool.ml.utils import patterns_type
from cftool.ml.utils import Comparer
from cftool.ml.utils import Estimator
from cftool.ml.utils import ModelPattern
from cftool.ml.utils import EnsemblePattern
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import TaskTypes

from ..dist import Task
from ..dist import Experiments
from ..types import data_type
from ..trainer import Trainer
from ..trainer import IntermediateResults
from ..pipeline import Pipeline
from ..protocol import DataProtocol
from ..misc.toolkit import to_2d


def make(model: str = "fcnn", **kwargs: Any) -> Pipeline:
    kwargs["model"] = model
    return Pipeline.make(kwargs)


SAVING_DELIM = "^_^"
pipelines_type = Union[
    Pipeline,
    List[Pipeline],
    Dict[str, Pipeline],
    Dict[str, List[Pipeline]],
]


def _to_saving_path(identifier: str, saving_folder: Optional[str]) -> str:
    if saving_folder is None:
        saving_path = identifier
    else:
        saving_path = os.path.join(saving_folder, identifier)
    return saving_path


def _make_saving_path(
    i: int,
    name: str,
    saving_path: str,
    remove_existing: bool,
) -> str:
    saving_path = os.path.abspath(saving_path)
    saving_folder, identifier = os.path.split(saving_path)
    postfix = f"{SAVING_DELIM}{name}{SAVING_DELIM}{i:04d}"
    if os.path.isdir(saving_folder) and remove_existing:
        for existing_model in os.listdir(saving_folder):
            if os.path.isdir(os.path.join(saving_folder, existing_model)):
                continue
            if existing_model.startswith(f"{identifier}{postfix}"):
                print(
                    f"{LoggingMixin.warning_prefix}"
                    f"'{existing_model}' was found, it will be removed"
                )
                os.remove(os.path.join(saving_folder, existing_model))
    return f"{saving_path}{postfix}"


def _to_pipelines(pipelines: pipelines_type) -> Dict[str, List[Pipeline]]:
    if isinstance(pipelines, dict):
        pipeline_dict = {}
        for key, value in pipelines.items():
            if isinstance(value, list):
                pipeline_dict[key] = value
            else:
                pipeline_dict[key] = [value]
    else:
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        pipeline_dict = {}
        for pipeline in pipelines:
            assert pipeline.model is not None
            key = pipeline.model.__identifier__
            pipeline_dict.setdefault(key, []).append(pipeline)
    return pipeline_dict


def evaluate(
    x: data_type,
    y: data_type = None,
    *,
    contains_labels: bool = True,
    pipelines: Optional[pipelines_type] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    metric_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    other_patterns: Optional[Dict[str, patterns_type]] = None,
    comparer_verbose_level: Optional[int] = 1,
) -> Comparer:
    if not contains_labels:
        msg = "`cflearn.evaluate` must be called with `contains_labels = True`"
        raise ValueError(msg)

    patterns = {}
    if pipelines is None:
        if y is None:
            raise ValueError("either `pipelines` or `y` should be provided")
        if metrics is None:
            raise ValueError("either `pipelines` or `metrics` should be provided")
        if metric_configs is None:
            metric_configs = [{} for _ in range(len(metrics))]
        if other_patterns is None:
            raise ValueError(
                "either `pipelines` or `other_patterns` should be provided"
            )
    else:
        pipelines = _to_pipelines(pipelines)
        if predict_config is None:
            predict_config = {}
        predict_config.setdefault("contains_labels", contains_labels)
        for name, pipeline_list in pipelines.items():
            first_pipeline = pipeline_list[0]
            data = first_pipeline.data
            if y is not None:
                y = to_2d(y)
            else:
                if not isinstance(x, str):
                    raise ValueError("`x` should be str when `y` is not provided")
                x, y = data.read_file(x, contains_labels=contains_labels)
                y = data.transform(x, y).y
            if metrics is None:
                metrics = [
                    k
                    for k, v in first_pipeline.trainer.metrics.items()
                    if v is not None
                ]
            if metric_configs is None:
                metric_configs = [
                    v.config
                    for k, v in first_pipeline.trainer.metrics.items()
                    if v is not None
                ]
            patterns[name] = [
                pipeline.to_pattern(**predict_config) for pipeline in pipeline_list
            ]
    if other_patterns is not None:
        for other_name in other_patterns.keys():
            if other_name in patterns:
                print(
                    f"{LoggingMixin.warning_prefix}'{other_name}' is found in "
                    "`other_patterns`, it will be overwritten"
                )
        update_dict(other_patterns, patterns)

    if isinstance(metrics, list):
        metrics_list = metrics
    else:
        assert isinstance(metrics, str)
        metrics_list = [metrics]
    if isinstance(metric_configs, list):
        metric_configs_list = metric_configs
    else:
        assert isinstance(metric_configs, dict)
        metric_configs_list = [metric_configs]

    estimators = [
        Estimator(metric, metric_config=metric_config)
        for metric, metric_config in zip(metrics_list, metric_configs_list)
    ]
    comparer = Comparer(patterns, estimators)
    comparer.compare(x, y, verbose_level=comparer_verbose_level)
    return comparer


def save(
    pipelines: pipelines_type,
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
    *,
    retain_data: bool = True,
) -> Dict[str, List[Pipeline]]:
    pipeline_dict = _to_pipelines(pipelines)
    saving_path = _to_saving_path(identifier, saving_folder)
    for name, pipeline_list in pipeline_dict.items():
        for i, pipeline in enumerate(pipeline_list):
            pipeline.save(
                _make_saving_path(i, name, saving_path, True),
                retain_data=retain_data,
                compress=True,
            )
    return pipeline_dict


def _fetch_saving_paths(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    saving_path = _to_saving_path(identifier, saving_folder)
    saving_path = os.path.abspath(saving_path)
    base_folder = os.path.dirname(saving_path)
    for existing_model in os.listdir(base_folder):
        if not os.path.isfile(os.path.join(base_folder, existing_model)):
            continue
        existing_model, existing_extension = os.path.splitext(existing_model)
        if existing_extension != ".zip":
            continue
        if SAVING_DELIM in existing_model:
            *folder, name, i = existing_model.split(SAVING_DELIM)
            if os.path.join(base_folder, SAVING_DELIM.join(folder)) != saving_path:
                continue
            new_path = _make_saving_path(int(i), name, saving_path, False)
            paths.setdefault(name, []).append(new_path)
    return paths


def load(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, List[Pipeline]]:
    paths = _fetch_saving_paths(identifier, saving_folder)
    pipelines = {
        k: [Pipeline.load(v, compress=True) for v in v_list]
        for k, v_list in paths.items()
    }
    if not pipelines:
        raise ValueError(
            f"'{identifier}' models not found with `saving_folder`={saving_folder}"
        )
    return pipelines


def _remove(identifier: str = "cflearn", saving_folder: str = None) -> None:
    for path_list in _fetch_saving_paths(identifier, saving_folder).values():
        for path in path_list:
            path = f"{path}.zip"
            print(f"{LoggingMixin.info_prefix}removing {path}...")
            os.remove(path)


def _rmtree(folder: str, patience: float = 10.0) -> None:
    if not os.path.isdir(folder):
        return None
    t = time.time()
    while True:
        try:
            if time.time() - t >= patience:
                print(f"\n{LoggingMixin.warning_prefix}failed to rmtree: {folder}")
                break
            shutil.rmtree(folder)
            break
        except:
            print("", end=".", flush=True)
            time.sleep(1)


def load_task(task: Task) -> Pipeline:
    return next(iter(load(saving_folder=task.saving_folder).values()))[0]


def transform_experiments(experiments: Experiments) -> Dict[str, List[Pipeline]]:
    return {k: list(map(load_task, v)) for k, v in experiments.tasks.items()}


class RepeatResult(NamedTuple):
    data: Optional[DataProtocol]
    experiments: Optional[Experiments]
    pipelines: Optional[Dict[str, List[Pipeline]]]
    patterns: Optional[Dict[str, List[ModelPattern]]]

    @property
    def trainers(self) -> Optional[Dict[str, List[Trainer]]]:
        if self.pipelines is None:
            return None
        return {k: [m.trainer for m in v] for k, v in self.pipelines.items()}

    @property
    def final_results(self) -> Optional[Dict[str, List[IntermediateResults]]]:
        trainers = self.trainers
        if trainers is None:
            return None
        final_results_dict: Dict[str, List[IntermediateResults]] = {}
        for k, v in trainers.items():
            local_results = final_results_dict.setdefault(k, [])
            for trainer in v:
                final_results = trainer.final_results
                if final_results is None:
                    raise ValueError(f"training of `Trainer` ({trainer}) is corrupted")
                local_results.append(final_results)
        return final_results_dict


# If `x_cv` is not provided, then `KRandom` split will be performed
def repeat_with(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    models: Union[str, List[str]] = "fcnn",
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    identifiers: Optional[Union[str, List[str]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    temp_folder: str = "__tmp__",
    return_patterns: bool = True,
    use_tqdm: bool = True,
    **kwargs: Any,
) -> RepeatResult:

    if isinstance(models, str):
        models = [models]
    if identifiers is None:
        identifiers = models.copy()
    elif isinstance(identifiers, str):
        identifiers = [identifiers]

    kwargs = shallow_copy_dict(kwargs)
    kwargs.setdefault("trigger_logging", False)
    kwargs["verbose_level"] = 0

    if sequential is None:
        sequential = num_jobs <= 1
    if model_configs is None:
        model_configs = {}

    pipelines_dict: Optional[Dict[str, List[Pipeline]]] = None
    if sequential:
        experiments = None
        kwargs["use_tqdm"] = False

        if not return_patterns:
            print(
                f"{LoggingMixin.warning_prefix}`return_patterns` should be True "
                "when `sequential` is True, because patterns will always be generated"
            )
            return_patterns = True

        def get(i_: int, model_: str) -> Pipeline:
            kwargs_ = shallow_copy_dict(kwargs)
            assert model_configs is not None
            model_config = model_configs.setdefault(model_, {})
            kwargs_ = update_dict(shallow_copy_dict(model_config), kwargs_)
            logging_folder = os.path.join(temp_folder, model_, str(i_))
            kwargs_.setdefault("logging_folder", logging_folder)
            m = make(model_, **shallow_copy_dict(kwargs_))
            return m.fit(x, y, x_cv, y_cv)

        pipelines_dict = {}
        iterator = zip(models, identifiers)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(models), position=0)
        for model, identifier in iterator:
            local_pipelines = []
            sub_iterator = range(num_repeat)
            if use_tqdm:
                sub_iterator = tqdm(
                    sub_iterator,
                    total=num_repeat,
                    position=1,
                    leave=False,
                )
            for i in sub_iterator:
                local_pipelines.append(get(i, model))
            pipelines_dict[identifier] = local_pipelines
    else:
        if num_jobs <= 1:
            print(
                f"{LoggingMixin.warning_prefix}we suggest setting `sequential` "
                f"to True when `num_jobs` is {num_jobs}"
            )

        experiments = Experiments(temp_folder, overwrite=False)
        experiments.run(
            None,
            x,
            y,
            x_cv,
            y_cv,
            models=models,
            model_configs=model_configs,
            identifiers=identifiers,
            num_repeat=num_repeat,
            num_jobs=num_jobs,
            use_tqdm=use_tqdm,
            **shallow_copy_dict(kwargs),
        )
        if return_patterns:
            pipelines_dict = transform_experiments(experiments)

    patterns = None
    if return_patterns:
        assert pipelines_dict is not None
        if predict_config is None:
            predict_config = {}
        patterns = {
            model: [m.to_pattern(**predict_config) for m in pipelines]
            for model, pipelines in pipelines_dict.items()
        }

    data = None
    if patterns is not None:
        data = patterns[identifiers[0]][0].model.data

    return RepeatResult(data, experiments, pipelines_dict, patterns)


def tasks_to_pipelines(tasks: List[Task]) -> List[Pipeline]:
    return list(map(load_task, tasks))


def tasks_to_patterns(tasks: List[Task], **kwargs: Any) -> List[pattern_type]:
    pipelines = tasks_to_pipelines(tasks)
    return [m.to_pattern(**shallow_copy_dict(kwargs)) for m in pipelines]


def make_toy_model(
    model: str = "fcnn",
    config: Optional[Dict[str, Any]] = None,
    *,
    task_type: task_type_type = "reg",
    data_tuple: Optional[Tuple[data_type, data_type]] = None,
    cuda: Optional[Union[int, str]] = "cpu",
) -> Pipeline:
    if config is None:
        config = {}
    if data_tuple is not None:
        x, y = data_tuple
        assert isinstance(x, list)
    else:
        if parse_task_type(task_type) is TaskTypes.REGRESSION:
            x, y = [[0]], [[1]]
        else:
            x, y = [[0], [1]], [[1], [0]]
    data_tuple = x, y
    model_config = {}
    if model in ("fcnn", "tree_dnn"):
        model_config = {
            "pipe_configs": {
                "fcnn": {
                    "head": {
                        "hidden_units": [100],
                        "mapping_configs": {"dropout": 0.0, "batch_norm": False},
                    }
                }
            }
        }
    base_config = {
        "model": model,
        "model_config": model_config,
        "cv_split": 0.0,
        "trigger_logging": False,
        "min_epoch": 1,
        "num_epoch": 2,
        "max_epoch": 4,
        "optimizer": "sgd",
        "optimizer_config": {"lr": 0.01},
        "task_type": task_type,
        "data_config": {
            "valid_columns": list(range(len(x[0]))),
            "label_process_method": "identical",
        },
        "verbose_level": 0,
        "cuda": cuda,
    }
    updated = update_dict(config, base_config)
    return make(**updated).fit(*data_tuple)


__all__ = [
    "make",
    "save",
    "load",
    "evaluate",
    "load_task",
    "repeat_with",
    "tasks_to_pipelines",
    "tasks_to_patterns",
    "transform_experiments",
    "make_toy_model",
    "Task",
    "Experiments",
    "ModelPattern",
    "EnsemblePattern",
    "RepeatResult",
    "SAVING_DELIM",
    "_remove",
    "_rmtree",
]
