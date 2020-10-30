import os
import json
import time
import shutil

from typing import *
from tqdm import tqdm
from cftool.misc import timestamp
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
from cfdata.tabular import TabularData
from cfdata.tabular import TimeSeriesConfig
from optuna.trial import Trial

from ..dist import Task
from ..dist import Experiments
from ..types import data_type
from ..types import general_config_type
from ..misc.toolkit import to_2d
from ..trainer.core import Trainer
from ..trainer.core import IntermediateResults
from ..pipeline.core import Pipeline


def _parse_config(config: general_config_type) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)
    return shallow_copy_dict(config)


def make(
    model: str = "fcnn",
    *,
    use_amp: Optional[bool] = None,
    use_simplify_data: Optional[bool] = None,
    config: general_config_type = None,
    increment_config: general_config_type = None,
    delim: Optional[str] = None,
    task_type: Optional[task_type_type] = None,
    skip_first: Optional[bool] = None,
    cv_split: Optional[Union[float, int]] = None,
    min_epoch: Optional[int] = None,
    num_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_snapshot_num: Optional[int] = None,
    clip_norm: Optional[float] = None,
    ema_decay: Optional[float] = None,
    ts_config: Optional[TimeSeriesConfig] = None,
    aggregation: Optional[str] = None,
    aggregation_config: Optional[Dict[str, Any]] = None,
    ts_label_collator_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    read_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    metric_config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[str] = None,
    scheduler: Optional[str] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
    optimizers: Optional[Dict[str, Any]] = None,
    logging_file: Optional[str] = None,
    logging_folder: Optional[str] = None,
    trigger_logging: Optional[bool] = None,
    trial: Optional[Trial] = None,
    tracker_config: Optional[Dict[str, Any]] = None,
    cuda: Optional[Union[int, str]] = None,
    verbose_level: Optional[int] = None,
    use_timing_context: Optional[bool] = None,
    use_tqdm: Optional[bool] = None,
    **kwargs: Any,
) -> Pipeline:
    kwargs = shallow_copy_dict(kwargs)
    cfg, inc_cfg = map(_parse_config, [config, increment_config])
    update_dict(update_dict(inc_cfg, cfg), kwargs)
    # pipeline general
    kwargs["model"] = model
    if cv_split is not None:
        kwargs["cv_split"] = cv_split
    if use_tqdm is not None:
        kwargs["use_tqdm"] = use_tqdm
    if use_timing_context is not None:
        kwargs["use_timing_context"] = use_timing_context
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if ts_label_collator_config is not None:
        kwargs["ts_label_collator_config"] = ts_label_collator_config
    if data_config is None:
        data_config = {}
    if use_simplify_data is not None:
        data_config["simplify"] = use_simplify_data
    if ts_config is not None:
        data_config["time_series_config"] = ts_config
    if task_type is not None:
        data_config["task_type"] = task_type
    if read_config is None:
        read_config = {}
    if delim is not None:
        read_config["delim"] = delim
    if skip_first is not None:
        read_config["skip_first"] = skip_first
    kwargs["data_config"] = data_config
    kwargs["read_config"] = read_config
    sampler_config = kwargs.setdefault("sampler_config", {})
    if aggregation is not None:
        sampler_config["aggregation"] = aggregation
    if aggregation_config is not None:
        sampler_config["aggregation_config"] = aggregation_config
    if logging_folder is not None:
        if logging_file is None:
            logging_file = f"{model}_{timestamp()}.log"
        kwargs["logging_folder"] = logging_folder
        kwargs["logging_file"] = logging_file
    if trigger_logging is not None:
        kwargs["trigger_logging"] = trigger_logging
    # trainer general
    trainer_config = kwargs.setdefault("trainer_config", {})
    if use_amp is not None:
        trainer_config["use_amp"] = use_amp
    if min_epoch is not None:
        trainer_config["min_epoch"] = min_epoch
    if num_epoch is not None:
        trainer_config["num_epoch"] = num_epoch
    if max_epoch is not None:
        trainer_config["max_epoch"] = max_epoch
    if max_snapshot_num is not None:
        trainer_config["max_snapshot_num"] = max_snapshot_num
    if clip_norm is not None:
        trainer_config["clip_norm"] = clip_norm
    # model general
    if model_config is None:
        model_config = {}
    if ema_decay is not None:
        model_config["ema_decay"] = ema_decay
    kwargs["model_config"] = model_config
    # metrics
    metric_config_: Dict[str, Any] = {}
    if metric_config is not None:
        metric_config_ = metric_config
    elif metrics is not None:
        metric_config_["types"] = metrics
    if metric_config_:
        trainer_config["metric_config"] = metric_config_
    # optimizers
    if optimizers is not None:
        if optimizer is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer` is set to '{optimizer}' "
                f"but `optimizers` is provided, so `optimizer` will be ignored"
            )
        if optimizer_config is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer_config` is set to '{optimizer_config}' "
                f"but `optimizers` is provided, so `optimizer_config` will be ignored"
            )
    else:
        preset_optimizer = {}
        if optimizer is not None:
            if optimizer_config is None:
                optimizer_config = {}
            preset_optimizer = {
                "optimizer": optimizer,
                "optimizer_config": optimizer_config,
            }
        if scheduler is not None:
            if scheduler_config is None:
                scheduler_config = {}
            preset_optimizer.update(
                {"scheduler": scheduler, "scheduler_config": scheduler_config}
            )
        if preset_optimizer:
            optimizers = {"all": preset_optimizer}
    if optimizers is not None:
        trainer_config["optimizers"] = optimizers
    # misc
    other_kwargs = {
        "cuda": cuda,
        "trial": trial,
        "tracker_config": tracker_config,
    }
    if verbose_level is not None:
        other_kwargs["verbose_level"] = verbose_level
    return Pipeline(kwargs, **other_kwargs)  # type: ignore


SAVING_DELIM = "^_^"
pipelines_type = Union[Pipeline, List[Pipeline], Dict[str, Pipeline]]


def _to_saving_path(identifier: str, saving_folder: Optional[str]) -> str:
    if saving_folder is None:
        saving_path = identifier
    else:
        saving_path = os.path.join(saving_folder, identifier)
    return saving_path


def _make_saving_path(name: str, saving_path: str, remove_existing: bool) -> str:
    saving_path = os.path.abspath(saving_path)
    saving_folder, identifier = os.path.split(saving_path)
    postfix = f"{SAVING_DELIM}{name}"
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


def _to_pipelines(pipelines: pipelines_type) -> Dict[str, Pipeline]:
    if not isinstance(pipelines, dict):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        names: List[str] = [
            pipeline.model.__identifier__ for pipeline in pipelines  # type: ignore
        ]
        if len(set(names)) != len(pipelines):
            raise ValueError(
                "pipeline names are not provided "
                "but identical pipeline.model is detected"
            )
        pipelines = dict(zip(names, pipelines))
    return pipelines


def estimate(
    x: data_type,
    y: data_type = None,
    *,
    contains_labels: bool = False,
    pipelines: Optional[pipelines_type] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    other_patterns: Optional[Dict[str, patterns_type]] = None,
    comparer_verbose_level: Optional[int] = 1,
) -> Comparer:
    patterns = {}
    if pipelines is None:
        if y is None:
            raise ValueError("either `pipelines` or `y` should be provided")
        if metrics is None:
            raise ValueError("either `pipelines` or `metrics` should be provided")
        if other_patterns is None:
            raise ValueError(
                "either `pipelines` or `other_patterns` should be provided"
            )
    else:
        pipelines = _to_pipelines(pipelines)
        if predict_config is None:
            predict_config = {}
        predict_config.setdefault("contains_labels", contains_labels)
        for name, pipeline in pipelines.items():
            if y is not None:
                y = to_2d(y)
            else:
                x, y = pipeline.data.read_file(x, contains_labels=contains_labels)
                y = pipeline.data.transform(x, y).y
            if metrics is None:
                metrics = [
                    k for k, v in pipeline.trainer.metrics.items() if v is not None
                ]
            patterns[name] = pipeline.to_pattern(**predict_config)
    if other_patterns is not None:
        for other_name in other_patterns.keys():
            if other_name in patterns:
                prefix = LoggingMixin.warning_prefix
                print(
                    f"{prefix}'{other_name}' is found in `other_patterns`, it will be overwritten"
                )
        update_dict(other_patterns, patterns)

    if isinstance(metrics, list):
        metrics_list = metrics
    else:
        assert isinstance(metrics, str)
        metrics_list = [metrics]

    estimators = list(map(Estimator, metrics_list))
    comparer = Comparer(patterns, estimators)
    comparer.compare(x, y, verbose_level=comparer_verbose_level)
    return comparer


def save(
    pipelines: pipelines_type,
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
    *,
    retain_data: bool = True,
) -> Dict[str, Pipeline]:
    pipelines = _to_pipelines(pipelines)
    saving_path = _to_saving_path(identifier, saving_folder)
    for name, pipeline in pipelines.items():
        pipeline.save(
            _make_saving_path(name, saving_path, True),
            retain_data=retain_data,
            compress=True,
        )
    return pipelines


def _fetch_saving_paths(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, str]:
    paths = {}
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
            *folder, name = existing_model.split(SAVING_DELIM)
            if os.path.join(base_folder, SAVING_DELIM.join(folder)) != saving_path:
                continue
            paths[name] = _make_saving_path(name, saving_path, False)
    return paths


def load(
    identifier: str = "cflearn",
    saving_folder: Optional[str] = None,
) -> Dict[str, Pipeline]:
    paths = _fetch_saving_paths(identifier, saving_folder)
    pipelines = {k: Pipeline.load(v, compress=True) for k, v in paths.items()}
    if not pipelines:
        raise ValueError(
            f"'{identifier}' models not found with `saving_folder`={saving_folder}"
        )
    return pipelines


def _remove(identifier: str = "cflearn", saving_folder: str = None) -> None:
    for path in _fetch_saving_paths(identifier, saving_folder).values():
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
    return next(iter(load(saving_folder=task.saving_folder).values()))


def transform_experiments(experiments: Experiments) -> Dict[str, List[Pipeline]]:
    return {k: list(map(load_task, v)) for k, v in experiments.tasks.items()}


class RepeatResult(NamedTuple):
    data: Optional[TabularData]
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
    num_jobs: int = 4,
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
            temp_folder=temp_folder,
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
    base_config = {
        "model": model,
        "model_config": {
            "hidden_units": [100],
            "mapping_configs": {"dropout": 0.0, "batch_norm": False},
        },
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
    }
    updated = update_dict(config, base_config)
    return make(**updated).fit(*data_tuple)


__all__ = [
    "make",
    "save",
    "load",
    "estimate",
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
