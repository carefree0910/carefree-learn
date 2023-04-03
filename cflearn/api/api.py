import os
import sys
import json
import math
import shutil

import numpy as np

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import print_info
from cftool.misc import random_hash
from cftool.misc import print_warning
from cftool.misc import fix_float_to_length
from cftool.misc import get_latest_workspace
from cftool.misc import prepare_workspace_from
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from ..data import MLData
from ..types import data_type
from ..types import sample_weights_type
from ..types import states_callback_type
from ..schema import loss_dict
from ..schema import metric_dict
from ..schema import ILoss
from ..schema import DLConfig
from ..schema import MLConfig
from ..schema import IDLModel
from ..schema import IMetric
from ..schema import IDataLoader
from ..schema import DataProcessorConfig
from ..trainer import get_sorted_checkpoints
from ..pipeline import PackType
from ..pipeline import Pipeline
from ..pipeline import TrainingPipeline
from ..pipeline import MLTrainingPipeline
from ..pipeline import DLInferencePipeline
from ..pipeline import DLEvaluationPipeline
from ..pipeline import DLPipelineSerializer
from ..pipeline import IEvaluationPipeline
from ..constants import SCORES_FILE
from ..constants import DEFAULT_ZOO_TAG
from ..constants import CHECKPOINTS_FOLDER
from ..zoo.core import configs_root
from ..zoo.core import _parse_config
from ..zoo.core import DLZoo
from ..zoo.core import TPipeline
from ..dist.ml import Experiment
from ..dist.ml import ExperimentResults


TScoringFn = Callable[[List[float], float, float], float]
# metric_type -> pipeline_name -> Statistics
TEvaluations = Dict[str, Dict[str, "Statistics"]]
TChoices = Optional[List[Optional[Union[int, Set[int]]]]]


# general


class Statistics(NamedTuple):
    sign: float
    mean: float
    std: float
    score: float


class Evaluator:
    @classmethod
    def evaluate(
        cls,
        loader: IDataLoader,
        pipelines: Dict[str, Union[IEvaluationPipeline, List[IEvaluationPipeline]]],
        *,
        scoring_function: Union[str, TScoringFn] = "default",
        verbose: bool = True,
    ) -> TEvaluations:
        all_metric_statistics: TEvaluations = {}
        if not isinstance(scoring_function, str):
            scoring_fn = scoring_function
        else:
            scoring_fn = getattr(cls, f"_{scoring_function}_scoring")
        for name in sorted(pipelines):
            ms = pipelines[name]
            if not isinstance(ms, list):
                ms = [ms]
            metric_outputs = [m.evaluate(loader) for m in ms]
            for metric_type in sorted(metric_outputs[0].metric_values):
                is_positive = set([o.is_positive[metric_type] for o in metric_outputs])
                if len(is_positive) != 1:
                    raise ValueError(
                        f"the `is_positive` property of metric '{metric_type}' "
                        "should be identical across different pipeline"
                    )
                sign = 1.0 if list(is_positive)[0] else -1.0
                metrics = [o.metric_values[metric_type] for o in metric_outputs]
                metrics_array = np.array(metrics, np.float64)
                mean, std = metrics_array.mean().item(), metrics_array.std().item()
                score = sign * scoring_fn(metrics, mean, std * sign)
                metric_statistics = all_metric_statistics.setdefault(metric_type, {})
                metric_statistics[name] = Statistics(sign, mean, std, score)
        if verbose:
            cls.report(all_metric_statistics)
        return all_metric_statistics

    @staticmethod
    def report(
        all_metric_statistics: TEvaluations,
        *,
        padding: int = 1,
        name_length: int = 16,
        float_length: int = 8,
        verbose: bool = True,
    ) -> str:
        # collect
        body: Dict[str, List[float]] = {}
        same_choices: TChoices = None
        best_choices: TChoices = None
        need_display_best_choice = False
        sub_header = sorted_names = None
        stat_types = ["mean", "std", "score"]
        sorted_metrics = sorted(all_metric_statistics)
        for metric_idx, metric_type in enumerate(sorted_metrics):
            statistics = all_metric_statistics[metric_type]
            if sorted_names is None:
                sorted_names = sorted(statistics)
                need_display_best_choice = len(sorted_names) > 1
            if sub_header is None:
                sub_header = stat_types * len(sorted_metrics)
            if best_choices is None and need_display_best_choice:
                same_choices = [None] * len(sub_header)
                best_choices = [None] * len(sub_header)
            for name_idx, name in enumerate(sorted_names):
                method_statistics = statistics.get(name)
                if method_statistics is None:
                    method_values = [math.nan for _ in stat_types]
                else:
                    method_values = [
                        getattr(method_statistics, stat_type)
                        for stat_type in stat_types
                    ]
                    if best_choices is not None:
                        for stat_idx, method_value in enumerate(method_values):
                            choice_idx = metric_idx * len(stat_types) + stat_idx
                            current_idx_choice = best_choices[choice_idx]
                            if current_idx_choice is None:
                                best_choices[choice_idx] = name_idx
                            else:
                                stat_type = stat_types[stat_idx]
                                chosen_stat = getattr(
                                    statistics[sorted_names[current_idx_choice]],  # type: ignore
                                    stat_type,
                                )
                                if method_value == chosen_stat:
                                    if same_choices[choice_idx] is None:  # type: ignore
                                        same_choices[choice_idx] = {name_idx}  # type: ignore
                                    else:
                                        same_choices[choice_idx].add(name_idx)  # type: ignore
                                elif stat_type == "std":
                                    if method_value < chosen_stat:
                                        same_choices[choice_idx] = None  # type: ignore
                                        best_choices[choice_idx] = name_idx
                                elif stat_type == "score":
                                    if method_value > chosen_stat:
                                        same_choices[choice_idx] = None  # type: ignore
                                        best_choices[choice_idx] = name_idx
                                elif stat_type == "mean":
                                    sign = method_statistics.sign
                                    if method_value * sign > chosen_stat * sign:
                                        same_choices[choice_idx] = None  # type: ignore
                                        best_choices[choice_idx] = name_idx
                                else:
                                    msg = f"unrecognized `stat_type` ('{stat_type}') occurred"
                                    raise ValueError(msg)
                body.setdefault(name, []).extend(method_values)
        # organize
        padding = 2 * (padding + 3)
        name_length = name_length + padding
        cell_length = float_length + padding
        num_statistic_types = len(stat_types)
        metric_type_length = num_statistic_types * cell_length + 2
        header_msg = (
            f"|{'metrics':^{name_length}s}|"
            + "|".join(
                [
                    f"{metric_type:^{metric_type_length}s}"
                    for metric_type in sorted_metrics
                ]
            )
            + "|"
        )
        subs = [f"{sub_header_item:^{cell_length}s}" for sub_header_item in sub_header]  # type: ignore
        sub_header_msg = f"|{' ' * name_length}|" + "|".join(subs) + "|"
        body_msgs = []
        for name_idx, name in enumerate(sorted_names):  # type: ignore
            cell_msgs = []
            for cell_idx, cell_item in enumerate(body[name]):
                cell_str = fix_float_to_length(cell_item, float_length)
                if best_choices is not None and (
                    best_choices[cell_idx] == name_idx
                    or (
                        same_choices[cell_idx] is not None  # type: ignore
                        and name_idx in same_choices[cell_idx]  # type: ignore
                    )
                ):
                    cell_str = f" -- {cell_str} -- "
                else:
                    cell_str = f"{cell_str:^{cell_length}s}"
                cell_msgs.append(cell_str)
            body_msgs.append(f"|{name:^{name_length}s}|" + "|".join(cell_msgs) + "|")
        msgs = [header_msg, sub_header_msg] + body_msgs
        length = len(body_msgs[0])
        single_split = "-" * length
        double_split = "=" * length
        main_msg = f"\n{single_split}\n".join(msgs)
        final_msg = f"{double_split}\n{main_msg}\n{double_split}"
        if verbose:
            print(final_msg)
        return final_msg

    # internal

    @classmethod
    def _default_scoring(cls, metrics: List[float], mean: float, std: float) -> float:
        return mean - std

    @classmethod
    def _mean_scoring(cls, metrics: List[float], mean: float, std: float) -> float:
        return mean

    @classmethod
    def _std_scoring(cls, metrics: List[float], mean: float, std: float) -> float:
        return mean + std


def evaluate(
    loader: IDataLoader,
    pipelines: Dict[str, Union[IEvaluationPipeline, List[IEvaluationPipeline]]],
    *,
    scoring_function: Union[str, TScoringFn] = "default",
    verbose: bool = True,
) -> TEvaluations:
    return Evaluator.evaluate(
        loader,
        pipelines,
        scoring_function=scoring_function,
        verbose=verbose,
    )


# dl


def _rewrite(path: str, prefix: str, temp_folder: Optional[str]) -> str:
    with open(path, "r") as rf:
        original_scripts = rf.read()
    if temp_folder is None:
        temp_folder = os.path.split(os.path.abspath(path))[0]
    tmp_path = os.path.join(temp_folder, f"{random_hash()}.py")
    with open(tmp_path, "w") as wf:
        wf.write(f"{prefix}\n\n{original_scripts}")
    print_info(f"`{tmp_path}` will be executed")
    return tmp_path


def run_accelerate(
    path: str,
    *,
    set_config: bool = True,
    workspace: str = "_ddp",
    temp_folder: Optional[str] = None,
    **kwargs: Any,
) -> None:
    def _convert_config() -> str:
        return " ".join([f"--{k}={v}" for k, v in kwargs.items()])

    # https://github.com/pytorch/pytorch/issues/37377
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    if set_config:
        os.system("accelerate config")
    tmp_path = _rewrite(
        path,
        f"""
from cflearn.misc.toolkit import _set_environ_workspace

_set_environ_workspace("{workspace}")
    """,
        temp_folder,
    )
    os.system(f"accelerate launch {_convert_config()} {tmp_path}")
    os.remove(tmp_path)


def run_multiple(
    path: str,
    model_name: str,
    cuda_list: Optional[List[Union[int, str]]],
    *,
    num_jobs: int = 2,
    num_multiple: int = 5,
    workplace: str = "_multiple",
    resource_config: Optional[Dict[str, Any]] = None,
    is_fix: bool = False,
    task_meta_fn: Optional[Callable[[int], Any]] = None,
    temp_folder: Optional[str] = None,
) -> None:
    def is_buggy(i_: int) -> bool:
        i_workplace = os.path.join(workplace, model_name, str(i_))
        i_latest_workplace = get_latest_workspace(i_workplace)
        if i_latest_workplace is None:
            return True
        checkpoint_folder = os.path.join(i_latest_workplace, CHECKPOINTS_FOLDER)
        if not os.path.isfile(os.path.join(checkpoint_folder, SCORES_FILE)):
            return True
        if not get_sorted_checkpoints(checkpoint_folder):
            return True
        return False

    if num_jobs <= 1:
        raise ValueError("`num_jobs` should greater than 1")
    # remove workplace if exists
    if os.path.isdir(workplace) and not is_fix:
        print_warning(f"'{workplace}' already exists, it will be erased")
        shutil.rmtree(workplace)
    tmp_path = _rewrite(
        path,
        """
import os
from cflearn.misc.toolkit import _set_environ_workspace
from cflearn.dist.ml.runs._utils import get_info
from cflearn.parameters import OPT

info = get_info(requires_data=False)
os.environ["CUDA_VISIBLE_DEVICES"] = str(info.meta["cuda"])
_set_environ_workspace(info.meta["workplace"])
OPT.meta_settings = info.meta
""",
        temp_folder,
    )
    # construct & execute an Experiment
    cudas = None if cuda_list is None else list(map(int, cuda_list))
    experiment = Experiment(
        num_jobs=num_jobs,
        available_cuda_list=cudas,
        resource_config=resource_config,
    )
    for i in range(num_multiple):
        if is_fix and not is_buggy(i):
            continue
        if task_meta_fn is None:
            i_meta_kw = {}
        else:
            i_meta_kw = task_meta_fn(i)
        if not is_fix:
            workplace_key = None
        else:
            workplace_key = model_name, str(i)
        experiment.add_task(
            model=model_name,
            root_workspace=workplace,
            workspace_key=workplace_key,
            run_command=f"{sys.executable} {tmp_path}",
            task_meta_kwargs=i_meta_kw,
        )
    experiment.run_tasks(use_tqdm=False)
    os.remove(tmp_path)


def save(m: Pipeline, folder: str, *, compress: bool = False) -> None:
    DLPipelineSerializer.save(m, folder, compress=compress)


def pack(
    workspace: str,
    export_folder: str,
    *,
    pack_type: PackType = PackType.INFERENCE,
    compress: bool = True,
) -> None:
    return DLPipelineSerializer.pack(
        workspace,
        export_folder,
        pack_type=pack_type,
        compress=compress,
    )


def pack_onnx(
    workplace: str,
    export_file: str = "model.onnx",
    dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
    *,
    input_sample: Optional[tensor_dict_type] = None,
    loader_sample: Optional[IDataLoader] = None,
    opset: int = 11,
    simplify: bool = True,
    num_samples: Optional[int] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> DLInferencePipeline:
    return DLPipelineSerializer.pack_onnx(
        workplace,
        export_file,
        dynamic_axes,
        input_sample=input_sample,
        loader_sample=loader_sample,
        opset=opset,
        simplify=simplify,
        num_samples=num_samples,
        verbose=verbose,
        **kwargs,
    )


def pack_scripted(workplace: str, export_file: str = "model.pt") -> DLInferencePipeline:
    return DLPipelineSerializer.pack_scripted(workplace, export_file)


def fuse_inference(
    src_folders: List[str],
    *,
    cuda: Optional[str] = None,
    num_picked: Optional[Union[int, float]] = None,
    states_callback: states_callback_type = None,
) -> DLInferencePipeline:
    return DLPipelineSerializer.fuse_inference(
        src_folders,
        cuda=cuda,
        num_picked=num_picked,
        states_callback=states_callback,
    )


def fuse_evaluation(
    src_folders: List[str],
    *,
    cuda: Optional[str] = None,
    num_picked: Optional[Union[int, float]] = None,
    states_callback: states_callback_type = None,
) -> DLEvaluationPipeline:
    return DLPipelineSerializer.fuse_evaluation(
        src_folders,
        cuda=cuda,
        num_picked=num_picked,
        states_callback=states_callback,
    )


def load_training(folder: str) -> TrainingPipeline:
    return DLPipelineSerializer.load_training(folder)


def load_inference(folder: str) -> DLInferencePipeline:
    return DLPipelineSerializer.load_inference(folder)


def load_evaluation(folder: str) -> DLEvaluationPipeline:
    return DLPipelineSerializer.load_evaluation(folder)


def make_model(name: str, **kwargs: Any) -> IDLModel:
    return IDLModel.make(name, kwargs)


def make_loss(name: str, **kwargs: Any) -> ILoss:
    return ILoss.make(name, kwargs)


def make_metric(name: str, **kwargs: Any) -> IMetric:
    return IMetric.make(name, kwargs)


def supported_losses() -> List[str]:
    return sorted(loss_dict)


def supported_metrics() -> List[str]:
    return sorted(metric_dict)


# ml


def _make_ml_data(
    x_train: Union[data_type, MLData],
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    train_others: Optional[np_dict_type] = None,
    valid_others: Optional[np_dict_type] = None,
    processor_config: Optional[DataProcessorConfig] = None,
) -> MLData:
    if isinstance(x_train, MLData):
        return x_train
    data = MLData.init(processor_config=processor_config)
    return data.fit(x_train, y_train, x_valid, y_valid, train_others, valid_others)


def fit_ml(
    x_train: Union[data_type, MLData],
    y_train: data_type = None,
    x_valid: data_type = None,
    y_valid: data_type = None,
    *,
    # pipeline
    config: Optional[DLConfig] = None,
    debug: bool = False,
    # data
    train_others: Optional[np_dict_type] = None,
    valid_others: Optional[np_dict_type] = None,
    processor_config: Optional[DataProcessorConfig] = None,
    # fit
    sample_weights: sample_weights_type = None,
    cuda: Optional[Union[int, str]] = None,
) -> MLTrainingPipeline:
    valid_config = (config or DLConfig()).copy()
    if debug:
        valid_config.to_debug()
    data = _make_ml_data(
        x_train,
        y_train,
        x_valid,
        y_valid,
        train_others,
        valid_others,
        processor_config,
    )
    fit_kwargs = dict(sample_weights=sample_weights, cuda=cuda)
    return MLTrainingPipeline.init(valid_config).fit(data, **fit_kwargs)  # type: ignore


def repeat_ml(
    data: MLData,
    config: DLConfig,
    num_repeat: int = 1,
    *,
    workspace: str = "_repeat",
    add_sub_workspace: bool = True,
    models: Optional[List[str]] = None,
    num_jobs: int = 1,
    use_cuda: bool = True,
    available_cuda_list: Optional[List[int]] = None,
    resource_config: Optional[Dict[str, Any]] = None,
) -> ExperimentResults:
    if add_sub_workspace:
        workspace = prepare_workspace_from(workspace)
    elif os.path.isdir(workspace):
        raise ValueError(f"workspace '{workspace}' already exists")
    config.sanity_check()
    experiment = Experiment(
        num_jobs=num_jobs,
        use_cuda=use_cuda,
        available_cuda_list=available_cuda_list,
        resource_config=resource_config,
    )
    data_folder = experiment.dump_data(data, workspace=workspace)
    if models is None:
        models = [config.model_name]
    task_kw = dict(root_workspace=workspace, config=config, data_folder=data_folder)
    for model in models:
        for _ in range(num_repeat):
            experiment.add_task(model=model, **task_kw)  # type: ignore
    return experiment.run_tasks()


def load_pipelines(results: ExperimentResults) -> Dict[str, List[DLEvaluationPipeline]]:
    pipelines: Dict[str, List[DLEvaluationPipeline]] = {}
    for workspace, workspace_key in zip(results.workspaces, results.workspace_keys):
        model = workspace_key[0]
        pipeline_folder = os.path.join(workspace, "pipeline")
        if os.path.isdir(pipeline_folder):
            pipelines.setdefault(model, []).append(load_evaluation(pipeline_folder))
    return pipelines


def make_toy_ml_model(
    model: str = "fcnn",
    config: Optional[MLConfig] = None,
    *,
    is_classification: bool = False,
    data_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cuda: Optional[str] = None,
    **kwargs: Any,
) -> MLTrainingPipeline:
    if config is None:
        config = MLConfig()
    # data
    if data_tuple is not None:
        x_np, y_np = data_tuple
        if is_classification:
            output_dim = len(np.unique(y_np))
        else:
            output_dim = y_np.shape[1]
    else:
        if not is_classification:
            x, y = [[0.0]], [[1.0]]
            output_dim = 1
        else:
            x, y = [[0.0], [1.0]], [[1], [0]]
            output_dim = 2
        x_np, y_np = map(np.array, [x, y])
    # model
    model_config = config.model_config or {}
    model_config.update(dict(input_dim=x_np.shape[1], output_dim=output_dim))
    if model in ("fcnn", "tree_dnn"):
        model_config.update(dict(hidden_units=[100], batch_norm=False, dropout=0.0))
    # re-assign
    config.num_epoch = 2
    config.max_epoch = 4
    config.model_name = model
    config.model_config = model_config
    config.loss_name = "focal" if is_classification else "mae"
    for k, v in kwargs.items():
        setattr(config, k, v)
    # fit & return
    m = MLTrainingPipeline.init(config)
    data = MLData.init().fit(x_np, y_np)
    return m.fit(data, cuda=cuda)


# zoo


class ModelItem(NamedTuple):
    name: str
    requirements: Dict[str, Any]


def model_zoo(*, verbose: bool = False) -> List[ModelItem]:
    def _squeeze_requirements(req: Dict[str, Any], d: Dict[str, Any]) -> None:
        for k, v in req.items():
            kd = d.get(k)
            if kd is None:
                continue
            if isinstance(v, dict):
                _squeeze_requirements(v, kd)
                continue
            assert isinstance(v, list)
            pop_indices = []
            for i, vv in enumerate(v):
                if vv in kd:
                    pop_indices.append(i)
            for i in pop_indices[::-1]:
                v.pop(i)

    models = []
    for task in sorted(os.listdir(configs_root)):
        if task == "common_":
            continue
        task_folder = os.path.join(configs_root, task)
        for model in sorted(os.listdir(task_folder)):
            model_folder = os.path.join(task_folder, model)
            for config_file in sorted(os.listdir(model_folder)):
                config_path = os.path.join(model_folder, config_file)
                d = _parse_config(config_path)
                requirements = d.pop("__requires__", {})
                _squeeze_requirements(requirements, d)
                tag = os.path.splitext(config_file)[0]
                name = f"{task}/{model}"
                if tag != DEFAULT_ZOO_TAG:
                    name = f"{name}.{tag}"
                models.append(ModelItem(name, requirements))
    if verbose:

        def _stringify_item(item: ModelItem) -> str:
            return f"{item.name:>{span}s}   |   {json.dumps(item.requirements)}"

        span = 42
        print(
            "\n".join(
                [
                    "=" * 120,
                    f"{'Names':>{span}s}   |   Requirements",
                    "-" * 120,
                    "\n".join(map(_stringify_item, models)),
                    "-" * 120,
                ]
            )
        )
    return models


def from_zoo(model: str, **kwargs: Any) -> TPipeline:
    return DLZoo.load_pipeline(model, **kwargs)
