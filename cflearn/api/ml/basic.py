import os
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from tqdm.autonotebook import tqdm
from cftool.ml import ModelPattern
from cftool.ml import EnsemblePattern
from cftool.dist import Parallel
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.ml.utils import patterns_type
from cftool.ml.utils import Comparer
from cftool.ml.utils import Estimator

from .pipeline import SimplePipeline
from .pipeline import CarefreePipeline
from ..basic import make
from ...types import data_type
from ...constants import WARNING_PREFIX
from ...constants import ML_PIPELINE_SAVE_NAME
from ...dist.ml import Experiment
from ...dist.ml import ExperimentResults
from ...misc.toolkit import to_2d
from ...misc.toolkit import get_latest_workplace
from ...misc.internal_ import MLData
from ...models.ml.protocol import MLCoreProtocol


def register_core(name: str) -> Callable[[Type], Type]:
    return MLCoreProtocol.register(name)


pipelines_type = Dict[str, List[SimplePipeline]]
various_pipelines_type = Union[
    SimplePipeline,
    List[SimplePipeline],
    Dict[str, SimplePipeline],
    pipelines_type,
]


def _to_pipelines(pipelines: various_pipelines_type) -> pipelines_type:
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
    pipelines: Optional[various_pipelines_type] = None,
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
            if not isinstance(data_pipeline, CarefreePipeline):
                raise ValueError("only `CarefreePipeline` can handle file inputs")
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
                    f"{WARNING_PREFIX}'{other_name}' is found in "
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


def task_loader(
    workplace: str,
    pipeline_base: Type[SimplePipeline] = CarefreePipeline,
    compress: bool = True,
) -> SimplePipeline:
    export_folder = os.path.join(workplace, ML_PIPELINE_SAVE_NAME)
    m = pipeline_base.load(export_folder=export_folder, compress=compress)
    assert isinstance(m, SimplePipeline)
    return m


def load_experiment_results(
    results: ExperimentResults,
    pipeline_base: Type[SimplePipeline],
) -> pipelines_type:
    pipelines_dict: Dict[str, Dict[int, SimplePipeline]] = {}
    iterator = list(zip(results.workplaces, results.workplace_keys))
    for workplace, workplace_key in tqdm(iterator, desc="load"):
        pipeline = task_loader(workplace, pipeline_base)
        model, str_i = workplace_key
        pipelines_dict.setdefault(model, {})[int(str_i)] = pipeline
    return {k: [v[i] for i in sorted(v)] for k, v in pipelines_dict.items()}


class RepeatResult(NamedTuple):
    data: Optional[MLData]
    experiment: Optional[Experiment]
    pipelines: Optional[Dict[str, List[SimplePipeline]]]
    patterns: Optional[Dict[str, List[ModelPattern]]]


def repeat_with(
    x: data_type,
    y: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
    pipeline_base: Type[SimplePipeline] = CarefreePipeline,
    workplace: str = "_repeat",
    models: Union[str, List[str]] = "fcnn",
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    return_patterns: bool = True,
    compress: bool = True,
    use_tqdm: bool = True,
    available_cuda_list: Optional[List[int]] = None,
    resource_config: Optional[Dict[str, Any]] = None,
    task_meta_kwargs: Optional[Dict[str, Any]] = None,
    is_fix: bool = False,
    **kwargs: Any,
) -> RepeatResult:
    if os.path.isdir(workplace) and not is_fix:
        print(f"{WARNING_PREFIX}'{workplace}' already exists, it will be erased")
        shutil.rmtree(workplace)
    kwargs = shallow_copy_dict(kwargs)
    if isinstance(models, str):
        models = [models]
    if sequential is None:
        sequential = num_jobs <= 1
    if model_configs is None:
        model_configs = {}

    def is_buggy(i_: int, model_: str) -> bool:
        i_workplace = os.path.join(workplace, model_, str(i_))
        return get_latest_workplace(i_workplace) is None

    def fetch_config(core_name: str) -> Dict[str, Any]:
        local_kwargs = shallow_copy_dict(kwargs)
        assert model_configs is not None
        local_core_config = model_configs.setdefault(core_name, {})
        local_kwargs["core_name"] = core_name
        local_kwargs["core_config"] = shallow_copy_dict(local_core_config)
        return shallow_copy_dict(local_kwargs)

    pipelines_dict: Optional[Dict[str, List[SimplePipeline]]] = None
    if sequential:
        cuda = kwargs.pop("cuda", None)
        experiment = None
        tqdm_settings = kwargs.setdefault("tqdm_settings", {})
        tqdm_settings["tqdm_position"] = 2
        if not return_patterns:
            print(
                f"{WARNING_PREFIX}`return_patterns` should be "
                "True when `sequential` is True, because patterns "
                "will always be generated"
            )
            return_patterns = True
        pipelines_dict = {}
        if not use_tqdm:
            iterator = models
        else:
            iterator = tqdm(models, total=len(models), position=0)
        for model in iterator:
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
                if is_fix and not is_buggy(i, model):
                    continue
                local_config = fetch_config(model)
                local_workplace = os.path.join(workplace, model, str(i))
                local_config.setdefault("workplace", local_workplace)
                m = pipeline_base(**local_config)
                m.fit(x, y, x_valid, y_valid, cuda=cuda)
                local_pipelines.append(m)
            pipelines_dict[model] = local_pipelines
    else:
        if num_jobs <= 1:
            print(
                f"{WARNING_PREFIX}we suggest setting `sequential` "
                f"to True when `num_jobs` is {num_jobs}"
            )
        # data
        data_folder = Experiment.dump_data_bundle(
            x,
            y,
            x_valid,
            y_valid,
            workplace=workplace,
        )
        # experiment
        experiment = Experiment(
            num_jobs=num_jobs,
            available_cuda_list=available_cuda_list,
            resource_config=resource_config,
        )
        for model in models:
            for i in range(num_repeat):
                if is_fix and not is_buggy(i, model):
                    continue
                local_config = fetch_config(model)
                experiment.add_task(
                    model=model,
                    compress=compress,
                    root_workplace=workplace,
                    workplace_key=(model, str(i)),
                    config=local_config,
                    data_folder=data_folder,
                    **(task_meta_kwargs or {}),
                )
        # finalize
        results = experiment.run_tasks(use_tqdm=use_tqdm)
        if return_patterns:
            pipelines_dict = load_experiment_results(results, pipeline_base)

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
        m = patterns[models[0]][0].model
        if isinstance(m, CarefreePipeline):
            data = m.data

    return RepeatResult(data, experiment, pipelines_dict, patterns)


def pack_repeat(
    workplace: str,
    pipeline_base: Type[SimplePipeline],
    *,
    num_jobs: int = 1,
) -> List[str]:
    sub_workplaces = []
    for stuff in sorted(os.listdir(workplace)):
        sub_workplaces.append(get_latest_workplace(os.path.join(workplace, stuff)))
    rs = Parallel(num_jobs).grouped(pipeline_base.pack, sub_workplaces).ordered_results
    return sum(rs, [])


def make_toy_model(
    model: str = "fcnn",
    config: Optional[Dict[str, Any]] = None,
    *,
    pipeline_type: str = "ml.carefree",
    is_classification: bool = False,
    data_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cuda: Optional[str] = None,
) -> SimplePipeline:
    if config is None:
        config = {}
    if data_tuple is not None:
        x_np, y_np = data_tuple
    else:
        if not is_classification:
            x, y = [[0]], [[1]]
        else:
            x, y = [[0], [1]], [[1], [0]]
        x_np, y_np = map(np.array, [x, y])
    model_config = {}
    if model in ("fcnn", "tree_dnn"):
        model_config = {
            "hidden_units": [100],
            "batch_norm": False,
            "dropout": 0.0,
        }
    base_config = {
        "core_name": model,
        "core_config": model_config,
        "is_classification": is_classification,
        "output_dim": 1 + int(is_classification),
        "valid_split": 0.0,
        "num_epoch": 2,
        "max_epoch": 4,
        "data_config": {
            "valid_columns": list(range(x_np.shape[1])),
            "label_process_method": "identical",
        },
    }
    updated = update_dict(config, base_config)
    m = make(pipeline_type, config=updated)
    assert isinstance(m, SimplePipeline)
    m.fit(x_np, y_np, cuda=cuda)
    return m


__all__ = [
    "register_core",
    "evaluate",
    "task_loader",
    "load_experiment_results",
    "repeat_with",
    "pack_repeat",
    "make_toy_model",
    "ModelPattern",
    "EnsemblePattern",
]
