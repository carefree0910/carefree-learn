import os

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import NamedTuple
from tqdm.autonotebook import tqdm
from cftool.ml import ModelPattern
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin
from cftool.ml.utils import patterns_type
from cftool.ml.utils import Comparer
from cftool.ml.utils import Estimator

from .pipeline import MLPipeline
from ...types import data_type
from ...dist.ml import Experiment
from ...misc.toolkit import to_2d
from ...misc.internal_ import MLData


pipelines_type = Union[
    MLPipeline,
    List[MLPipeline],
    Dict[str, MLPipeline],
    Dict[str, List[MLPipeline]],
]


def _to_pipelines(pipelines: pipelines_type) -> Dict[str, List[MLPipeline]]:
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
    metrics: Union[str, List[str]],
    metric_configs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    contains_labels: bool = True,
    pipelines: Optional[pipelines_type] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    other_patterns: Optional[Dict[str, patterns_type]] = None,
    comparer_verbose_level: Optional[int] = 1,
) -> Comparer:
    if not contains_labels:
        err_msg = "`cflearn.evaluate` must be called with `contains_labels = True`"
        raise ValueError(err_msg)

    if metric_configs is None:
        metric_configs = [{} for _ in range(len(metrics))]

    patterns = {}
    if pipelines is None:
        msg = None
        if y is None:
            msg = "either `pipelines` or `y` should be provided"
        if other_patterns is None:
            msg = "either `pipelines` or `other_patterns` should be provided"
        if msg is not None:
            raise ValueError(msg)
    else:
        pipelines = _to_pipelines(pipelines)
        # get data
        # TODO : different pipelines may have different labels
        if y is not None:
            y = to_2d(y)
        else:
            if not isinstance(x, str):
                raise ValueError("`x` should be str when `y` is not provided")
            data_pipeline = list(pipelines.values())[0][0]
            data = data_pipeline.data
            x, y = data.read_file(x, contains_labels=contains_labels)
            y = data.transform(x, y).y
        # get metrics
        if predict_config is None:
            predict_config = {}
        predict_config.setdefault("contains_labels", contains_labels)
        for name, pipeline_list in pipelines.items():
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


class RepeatResult(NamedTuple):
    data: Optional[MLData]
    experiment: Optional[Experiment]
    pipelines: Optional[Dict[str, List[MLPipeline]]]
    patterns: Optional[Dict[str, List[ModelPattern]]]


def repeat_with(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    workplace: str = "_repeat",
    core_names: Union[str, List[str]] = "fcnn",
    core_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    return_patterns: bool = True,
    compress: bool = True,
    use_tqdm: bool = True,
    **kwargs: Any,
) -> RepeatResult:
    kwargs = shallow_copy_dict(kwargs)
    if isinstance(core_names, str):
        core_names = [core_names]
    if sequential is None:
        sequential = num_jobs <= 1
    if core_configs is None:
        core_configs = {}

    def fetch_config(core_name: str) -> Dict[str, Any]:
        local_kwargs = shallow_copy_dict(kwargs)
        assert core_configs is not None
        local_core_config = core_configs.setdefault(core_name, {})
        local_kwargs["core_name"] = core_name
        local_kwargs["core_config"] = shallow_copy_dict(local_core_config)
        return shallow_copy_dict(local_kwargs)

    pipelines_dict: Optional[Dict[str, List[MLPipeline]]] = None
    if sequential:
        experiment = None
        tqdm_settings = kwargs.setdefault("tqdm_settings", {})
        tqdm_settings["tqdm_position"] = 2
        if not return_patterns:
            print(
                f"{LoggingMixin.warning_prefix}`return_patterns` should be "
                "True when `sequential` is True, because patterns "
                "will always be generated"
            )
            return_patterns = True
        pipelines_dict = {}
        if not use_tqdm:
            iterator = core_names
        else:
            iterator = tqdm(core_names, total=len(core_names), position=0)
        for core in iterator:
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
                local_config = fetch_config(core)
                local_workplace = os.path.join(workplace, core, str(i))
                local_config.setdefault("workplace", local_workplace)
                local_pipelines.append(MLPipeline(**local_config).fit(x, y, x_cv, y_cv))
            pipelines_dict[core] = local_pipelines
    else:
        if num_jobs <= 1:
            print(
                f"{LoggingMixin.warning_prefix}we suggest setting `sequential` "
                f"to True when `num_jobs` is {num_jobs}"
            )
        # data
        data_folder = Experiment.dump_data_bundle(x, y, x_cv, y_cv, workplace=workplace)
        # experiment
        experiment = Experiment(num_jobs=num_jobs)
        for core in core_names:
            for i in range(num_repeat):
                local_config = fetch_config(core)
                experiment.add_task(
                    core=core,
                    compress=compress,
                    root_workplace=workplace,
                    config=local_config,
                    data_folder=data_folder,
                )
        # finalize
        results = experiment.run_tasks(use_tqdm=use_tqdm)
        # TODO : fix here
        if return_patterns:
            pass

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
        data = patterns[core_names[0]][0].model.data

    return RepeatResult(data, experiment, pipelines_dict, patterns)


__all__ = [
    "evaluate",
    "repeat_with",
]
