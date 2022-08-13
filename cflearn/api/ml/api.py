import os
import json
import shutil

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from tqdm.autonotebook import tqdm
from cftool.dist import Parallel
from cftool.misc import update_dict
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import get_latest_workplace

from .pipeline import MLPipeline
from .pipeline import MLCarefreePipeline
from ...data import MLData
from ...data import MLInferenceData
from ...types import configs_type
from ...types import sample_weights_type
from ...trainer import get_sorted_checkpoints
from ...constants import SCORES_FILE
from ...constants import CHECKPOINTS_FOLDER
from ...constants import ML_PIPELINE_SAVE_NAME
from ...dist.ml import Experiment
from ...dist.ml import ExperimentResults
from ...misc.toolkit import to_2d

try:
    from cfdata.tabular import TabularData
except:
    TabularData = None
try:
    from cfml.misc.toolkit import patterns_type
    from cfml.misc.toolkit import Comparer
    from cfml.misc.toolkit import Estimator
    from cfml.misc.toolkit import ModelPattern
except:
    patterns_type = None
    Comparer = None
    Estimator = None
    ModelPattern = None


pipelines_type = Dict[str, List[MLPipeline]]
various_pipelines_type = Union[
    MLPipeline,
    List[MLPipeline],
    Dict[str, MLPipeline],
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
    data: MLInferenceData,
    *,
    metrics: Union[str, List[str]],
    metric_configs: configs_type = None,
    contains_labels: bool = True,
    pipelines: Optional[various_pipelines_type] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    other_patterns: Optional[Dict[str, patterns_type]] = None,
    comparer_verbose_level: Optional[int] = 1,
) -> Comparer:
    if Comparer is None:
        raise ValueError("`carefree-ml` is needed for `evaluate`")
    if not data.for_inference:
        raise ValueError("`data.for_inference` should be `True` in `evaluate`")

    if not contains_labels:
        err_msg = "`cflearn.evaluate` must be called with `contains_labels = True`"
        raise ValueError(err_msg)

    if metric_configs is None:
        metric_configs = [{} for _ in range(len(metrics))]

    patterns = {}
    x, y = data.x_train, data.y_train
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
            cf_data = data.cf_data
            assert cf_data is not None
            x, y = cf_data.read_file(x, contains_labels=contains_labels)
            y = cf_data.transform(x, y).y
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
                print_warning(
                    f"'{other_name}' is found in "
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
    comparer.compare(data, y, verbose_level=comparer_verbose_level)
    return comparer


def task_loader(
    workplace: str,
    pipeline_base: Type[MLPipeline] = MLCarefreePipeline,
    *,
    to_original_device: bool = False,
    compress: bool = True,
) -> MLPipeline:
    export_folder = os.path.join(workplace, ML_PIPELINE_SAVE_NAME)
    m = pipeline_base.load(
        export_folder=export_folder,
        to_original_device=to_original_device,
        compress=compress,
    )
    assert isinstance(m, MLPipeline)
    return m


def load_experiment_results(
    results: ExperimentResults,
    pipeline_base: Type[MLPipeline],
    to_original_device: bool = False,
) -> pipelines_type:
    pipelines_dict: Dict[str, Dict[int, MLPipeline]] = {}
    iterator = list(zip(results.workplaces, results.workplace_keys))
    for workplace, workplace_key in tqdm(iterator, desc="load"):
        pipeline = task_loader(
            workplace,
            pipeline_base,
            to_original_device=to_original_device,
        )
        model, str_i = workplace_key
        pipelines_dict.setdefault(model, {})[int(str_i)] = pipeline
    return {k: [v[i] for i in sorted(v)] for k, v in pipelines_dict.items()}


class RepeatResult(NamedTuple):
    data: Optional[TabularData]
    experiment: Optional[Experiment]
    pipelines: Optional[Dict[str, List[MLPipeline]]]
    patterns: Optional[Dict[str, List[ModelPattern]]]


def repeat_with(
    data: MLData,
    *,
    carefree: bool = True,
    workplace: str = "_repeat",
    models: Union[str, List[str]] = "fcnn",
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    predict_config: Optional[Dict[str, Any]] = None,
    sample_weights: sample_weights_type = None,
    sequential: Optional[bool] = None,
    num_jobs: int = 1,
    num_repeat: int = 5,
    return_patterns: bool = True,
    compress: bool = True,
    use_tqdm: bool = True,
    available_cuda_list: Optional[List[int]] = None,
    resource_config: Optional[Dict[str, Any]] = None,
    task_meta_kwargs: Optional[Dict[str, Any]] = None,
    to_original_device: bool = False,
    is_fix: bool = False,
    **kwargs: Any,
) -> RepeatResult:
    if os.path.isdir(workplace) and not is_fix:
        print_warning(f"'{workplace}' already exists, it will be erased")
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
        i_latest_workplace = get_latest_workplace(i_workplace)
        if i_latest_workplace is None:
            return True
        checkpoint_folder = os.path.join(i_latest_workplace, CHECKPOINTS_FOLDER)
        if not os.path.isfile(os.path.join(checkpoint_folder, SCORES_FILE)):
            return True
        if not get_sorted_checkpoints(checkpoint_folder):
            return True
        return False

    def fetch_config(core_name: str) -> Dict[str, Any]:
        local_kwargs = shallow_copy_dict(kwargs)
        assert model_configs is not None
        local_core_config = model_configs.setdefault(core_name, {})
        local_kwargs["core_name"] = core_name
        local_kwargs["core_config"] = shallow_copy_dict(local_core_config)
        return shallow_copy_dict(local_kwargs)

    pipeline_base = MLCarefreePipeline if carefree else MLPipeline
    pipelines_dict: Optional[Dict[str, List[MLPipeline]]] = None
    if sequential:
        cuda = kwargs.pop("cuda", None)
        experiment = None
        tqdm_settings = kwargs.setdefault("tqdm_settings", {})
        if tqdm_settings is None:
            kwargs["tqdm_settings"] = tqdm_settings = {}
        tqdm_settings["tqdm_position"] = 2
        if not return_patterns:
            print_warning(
                "`return_patterns` should be "
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
                m.fit(data, sample_weights=sample_weights, cuda=cuda)
                local_pipelines.append(m)
            pipelines_dict[model] = local_pipelines
    else:
        if num_jobs <= 1:
            print_warning(
                "we suggest setting `sequential` "
                f"to True when `num_jobs` is {num_jobs}"
            )
        # data
        data.prepare(sample_weights)
        data_folder = Experiment.dump_data(data, workplace=workplace)
        # experiment
        experiment = Experiment(
            num_jobs=num_jobs,
            available_cuda_list=available_cuda_list,
            resource_config=resource_config,
        )
        # meta
        meta = shallow_copy_dict(task_meta_kwargs or {})
        meta["carefree"] = carefree
        # add tasks
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
                    **shallow_copy_dict(meta),
                )
        # finalize
        results = experiment.run_tasks(use_tqdm=use_tqdm)
        if return_patterns:
            pipelines_dict = load_experiment_results(
                results,
                pipeline_base,
                to_original_device,
            )

    patterns = None
    if return_patterns:
        assert pipelines_dict is not None
        if predict_config is None:
            predict_config = {}
        patterns = {
            model: [m.to_pattern(**predict_config) for m in pipelines]
            for model, pipelines in pipelines_dict.items()
        }

    cf_data = None
    if patterns is not None:
        m = patterns[models[0]][0].model
        if isinstance(m, MLCarefreePipeline):
            cf_data = m.cf_data

    return RepeatResult(cf_data, experiment, pipelines_dict, patterns)


def pack_repeat(
    workplace: str,
    pipeline_base: Type[MLPipeline],
    *,
    num_jobs: int = 1,
) -> List[str]:
    sub_workplaces = []
    for stuff in sorted(os.listdir(workplace)):
        stuff_path = os.path.join(workplace, stuff)
        if not os.path.isdir(stuff_path):
            continue
        sub_workplaces.append(get_latest_workplace(stuff_path))
    rs = Parallel(num_jobs).grouped(pipeline_base.pack, sub_workplaces).ordered_results
    return sum(rs, [])


def pick_from_repeat_and_pack(
    workplace: str,
    pipeline_base: Type[MLPipeline],
    *,
    num_pick: int,
    num_jobs: int = 1,
) -> List[str]:
    score_workplace_pairs = []
    for stuff in sorted(os.listdir(workplace)):
        stuff_path = os.path.join(workplace, stuff)
        if not os.path.isdir(stuff_path):
            continue
        sub_workplace = get_latest_workplace(stuff_path)
        assert sub_workplace is not None, "internal error occurred"
        score_path = os.path.join(sub_workplace, CHECKPOINTS_FOLDER, SCORES_FILE)
        with open(score_path, "r") as f:
            score = float(max(json.load(f).values()))
            score_workplace_pairs.append((score, sub_workplace))
    score_workplace_pairs = sorted(score_workplace_pairs)[::-1]
    sub_workplaces = [pair[1] for pair in score_workplace_pairs[:num_pick]]
    rs = Parallel(num_jobs).grouped(pipeline_base.pack, sub_workplaces).ordered_results
    return sum(rs, [])


def make_toy_model(
    model: str = "fcnn",
    config: Optional[Dict[str, Any]] = None,
    *,
    is_classification: bool = False,
    cf_data_config: Optional[Dict[str, Any]] = None,
    data_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cuda: Optional[str] = None,
) -> MLPipeline:
    if config is None:
        config = {}
    if data_tuple is not None:
        x_np, y_np = data_tuple
        if is_classification:
            output_dim = len(np.unique(y_np))
        else:
            output_dim = y_np.shape[1]
    else:
        if not is_classification:
            x, y = [[0]], [[1.0]]
            output_dim = 1
        else:
            x, y = [[0], [1]], [[1], [0]]
            output_dim = 2
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
        "output_dim": output_dim,
        "num_epoch": 2,
        "max_epoch": 4,
    }
    updated = update_dict(config, base_config)
    pipelines_type = "ml" if cf_data_config is None else "ml.carefree"
    m = MLPipeline.make(pipelines_type, updated)
    if cf_data_config is None:
        data = MLData(
            x_np,
            y_np,
            is_classification=is_classification,
            valid_split=0.0,
        )
    else:
        cf_data_config = update_dict(
            cf_data_config,
            dict(
                valid_columns=list(range(x_np.shape[1])),
                label_process_method="identical",
            ),
        )
        data = MLData.with_cf_data(
            x_np,
            y_np,
            is_classification=is_classification,
            cf_data_config=cf_data_config,
            valid_split=0.0,
        )
    m.fit(data, cuda=cuda)
    return m


__all__ = [
    "evaluate",
    "task_loader",
    "load_experiment_results",
    "repeat_with",
    "pack_repeat",
    "pick_from_repeat_and_pack",
    "make_toy_model",
    "RepeatResult",
]
