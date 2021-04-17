from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cftool.misc import update_dict
from cftool.misc import LoggingMixin
from cftool.ml.utils import patterns_type
from cftool.ml.utils import Comparer
from cftool.ml.utils import Estimator

from .pipeline import MLPipeline
from ...types import data_type
from ...misc.toolkit import to_2d


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


__all__ = [
    "evaluate",
]
