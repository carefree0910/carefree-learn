import os

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cftool.ml.param_utils import *
from cfdata.tabular import *
from functools import partial
from cftool.ml.hpo import HPOBase
from cftool.ml import register_metric
from cfdata.tabular.processors.base import Processor

from .dist import *
from .bases import *
from .models import *
from .modules import *
from .misc.toolkit import eval_context, Initializer


# register

def register_initializer(name):
    def _register(f):
        Initializer.add_initializer(f, name)
        return f
    return _register


def register_processor(name):
    return Processor.register(name)


# API

def make(model: str = "fcnn",
         *,
         delim: str = None,
         skip_first: bool = None,
         cv_split: Union[float, int] = 0.1,
         min_epoch: int = None,
         num_epoch: int = None,
         max_epoch: int = None,
         batch_size: int = None,
         clip_norm: float = None,
         ema_decay: float = None,
         data_config: Dict[str, Any] = None,
         read_config: Dict[str, Any] = None,
         model_config: Dict[str, Any] = None,
         metrics: Union[str, List[str]] = None,
         metric_config: Dict[str, Any] = None,
         optimizer: str = None,
         scheduler: str = None,
         optimizer_config: Dict[str, Any] = None,
         scheduler_config: Dict[str, Any] = None,
         optimizers: Dict[str, Any] = None,
         logging_file: str = None,
         logging_folder: str = None,
         trigger_logging: bool = None,
         cuda: Union[int, str] = 0,
         verbose_level: int = 2,
         use_tqdm: bool = True,
         **kwargs) -> Wrapper:
    # wrapper general
    kwargs["model"] = model
    kwargs["cv_split"] = cv_split
    if data_config is not None:
        kwargs["data_config"] = data_config
    if read_config is None:
        read_config = {}
    if delim is not None:
        read_config["delim"] = delim
    if skip_first is not None:
        read_config["skip_first"] = skip_first
    kwargs["read_config"] = read_config
    if model_config is not None:
        kwargs["model_config"] = model_config
    if logging_folder is not None:
        if logging_file is None:
            logging_file = f"{model}_{timestamp()}.log"
        kwargs["logging_folder"] = logging_folder
        kwargs["logging_file"] = logging_file
    if trigger_logging is not None:
        kwargs["trigger_logging"] = trigger_logging
    # pipeline general
    pipeline_config = kwargs.setdefault("pipeline_config", {})
    pipeline_config["use_tqdm"] = use_tqdm
    if min_epoch is not None:
        pipeline_config["min_epoch"] = min_epoch
    if num_epoch is not None:
        pipeline_config["num_epoch"] = num_epoch
    if max_epoch is not None:
        pipeline_config["max_epoch"] = max_epoch
    if batch_size is not None:
        pipeline_config["batch_size"] = batch_size
    if clip_norm is not None:
        pipeline_config["clip_norm"] = clip_norm
    if ema_decay is not None:
        pipeline_config["ema_decay"] = ema_decay
    # metrics
    if metric_config is not None:
        if metrics is not None:
            print(
                f"{LoggingMixin.warning_prefix}`metrics` is set to '{metrics}' "
                f"but `metric_config` is provided, so `metrics` will be ignored")
    elif metrics is not None:
        metric_config = {"types": metrics}
    if metric_config is not None:
        pipeline_config["metric_config"] = metric_config
    # optimizers
    if optimizers is not None:
        if optimizer is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer` is set to '{optimizer}' "
                f"but `optimizers` is provided, so `optimizer` will be ignored")
        if optimizer_config is not None:
            print(
                f"{LoggingMixin.warning_prefix}`optimizer_config` is set to '{optimizer_config}' "
                f"but `optimizers` is provided, so `optimizer_config` will be ignored")
    else:
        preset_optimizer = {}
        if optimizer is not None:
            if optimizer_config is None:
                optimizer_config = {}
            preset_optimizer = {"optimizer": optimizer, "optimizer_config": optimizer_config}
        if scheduler is not None:
            if scheduler_config is None:
                scheduler_config = {}
            preset_optimizer.update({"scheduler": scheduler, "scheduler_config": scheduler_config})
        if preset_optimizer:
            optimizers = {"all": preset_optimizer}
    if optimizers is not None:
        pipeline_config["optimizers"] = optimizers
    return Wrapper(kwargs, cuda=cuda, verbose_level=verbose_level)


class EvaluateTransformer:
    def __init__(self, data: TabularData):
        self.data = data

    def get_xy(self,
               x: data_type,
               y: data_type = None) -> Tuple[data_type, data_type]:
        if y is None:
            x, y = self.data.read_file(x)
        y = self.data.transform_labels(y)
        return x, y


SAVING_DELIM = "^_^"
wrappers_dict_type = Dict[str, Wrapper]
wrappers_type = Union[Wrapper, List[Wrapper], wrappers_dict_type]
repeat_result_type = Tuple[EvaluateTransformer, Union[List[ModelPattern], Dict[str, List[ModelPattern]]]]


def _to_saving_path(identifier: str,
                    saving_folder: str) -> str:
    if saving_folder is None:
        saving_path = identifier
    else:
        saving_path = os.path.join(saving_folder, identifier)
    return saving_path


def _make_saving_path(name: str,
                      saving_path: str,
                      remove_existing: bool) -> str:
    saving_path = os.path.abspath(saving_path)
    saving_folder, identifier = os.path.split(saving_path)
    postfix = f"{SAVING_DELIM}{name}"
    if os.path.isdir(saving_folder) and remove_existing:
        for existing_model in os.listdir(saving_folder):
            if os.path.isdir(os.path.join(saving_folder, existing_model)):
                continue
            if existing_model.startswith(f"{identifier}{postfix}"):
                print(f"{LoggingMixin.warning_prefix}"
                      f"'{existing_model}' was found, it will be removed")
                os.remove(os.path.join(saving_folder, existing_model))
    return f"{saving_path}{postfix}"


def load_task(task: Task) -> Wrapper:
    return next(iter(load(saving_folder=task.saving_folder).values()))


def repeat_with(x: data_type,
                y: data_type = None,
                x_cv: data_type = None,
                y_cv: data_type = None,
                *,
                models: Union[str, List[str]] = "fcnn",
                identifiers: Union[str, List[str]] = None,
                num_repeat: int = 5,
                num_parallel: int = 4,
                temp_folder: str = "__tmp__",
                return_tasks: bool = False,
                use_tqdm: bool = True,
                **kwargs) -> Union[repeat_result_type, Dict[str, List[Task]]]:
    if isinstance(models, str):
        models = [models]
    if identifiers is None:
        identifiers = models.copy()
    elif isinstance(identifiers, str):
        identifiers = [identifiers]

    kwargs["trigger_logging"] = False
    kwargs["verbose_level"] = 0

    tasks = patterns = None
    if num_parallel == 0 or num_repeat == 1:
        kwargs.setdefault("use_tqdm", False)
        if return_tasks:
            tasks = {}
            for i in range(num_repeat):
                for model, identifier in zip(models, identifiers):
                    task = Task(i, model, identifier, temp_folder)
                    task.fit(make, save, x, y, x_cv, y_cv, **kwargs)
                    tasks.setdefault(identifier, []).append(task)
        else:
            patterns = {}
            for model, identifier in zip(models, identifiers):
                init_method = lambda: make(model, **kwargs)
                train_method = lambda m: m.fit(x, y, x_cv, y_cv)
                pattern_kwargs = {"init_method": init_method, "train_method": train_method}
                patterns[identifier] = ModelPattern.repeat(num_repeat, **pattern_kwargs)
    else:
        results = Experiments().run(
            make, save, load_task, x, y, x_cv, y_cv,
            models=models, identifiers=identifiers,
            num_repeat=num_repeat, num_parallel=num_parallel,
            return_tasks=return_tasks, use_tqdm=use_tqdm,
            temp_folder=temp_folder, **kwargs
        )
        if return_tasks:
            tasks = results
        else:
            patterns = {
                model: [ModelPattern(init_method=lambda: wrapper) for wrapper in wrappers]
                for model, wrappers in results.items()
            }

    if return_tasks:
        return tasks

    first_patterns = patterns[identifiers[0]]
    tr_data = first_patterns[0].model.tr_data
    if len(identifiers) == 1:
        patterns = first_patterns
    return EvaluateTransformer(tr_data), patterns


def ensemble(patterns: List[ModelPattern],
             ensemble_method: Union[str, collate_fn_type] = "default") -> EnsemblePattern:
    return EnsemblePattern(patterns, ensemble_method)


def tune_with(x: data_type,
              y: data_type = None,
              x_cv: data_type = None,
              y_cv: data_type = None,
              *,
              model: str = "fcnn",
              hpo_method: str = "bo",
              params: Dict[str, DataType] = None,
              task_type: TaskTypes = None,
              metrics: Union[str, List[str]] = None,
              num_jobs: int = None,
              num_repeat: int = 5,
              num_parallel: int = 4,
              num_search: int = 10,
              temp_folder: str = "__tmp__",
              score_weights: Union[Dict[str, float], None] = None,
              estimator_scoring_function: Union[str, scoring_fn_type] = "default",
              search_config: Dict[str, Any] = None,
              verbose_level: int = 2,
              **kwargs) -> HPOBase:

    if isinstance(x, str):
        read_config = kwargs.get("read_config", {})
        delim = read_config.get("delim", kwargs.get("delim"))
        if delim is not None:
            read_config["delim"] = delim
        else:
            print(
                f"{LoggingMixin.warning_prefix}delimiter of the given file dataset is not provided, "
                "this may cause incorrect parsing"
            )
        if y is not None:
            read_config["y"] = y
        tr_data = TabularData(task_type=task_type)
        tr_data.read(x, **read_config)
        y = tr_data.processed.y
        if x_cv is not None:
            if y_cv is None:
                y_cv = tr_data.transform(x_cv).y
            else:
                y_cv = tr_data.transform_labels(y_cv)
    elif y is None:
        raise ValueError("`x` should be a file when `y` is not provided")

    def _creator(x_, y_, params_) -> Dict[str, List[Task]]:
        base_params = shallow_copy_dict(kwargs)
        update_dict(params_, base_params)
        base_params["verbose_level"] = 0
        base_params["use_tqdm"] = False
        if isinstance(x_, str):
            y_ = y_cv_ = None
        else:
            y_cv_ = None if y_cv is None else y_cv.copy()
        num_parallel_ = num_parallel if hpo.is_sequential else 0
        return repeat_with(
            x_, y_, x_cv, y_cv_,
            num_repeat=num_repeat, num_parallel=num_parallel_,
            models=model, identifiers=hash_code(str(params_)),
            temp_folder=temp_folder, return_tasks=True, **base_params
        )

    def _converter(created: List[Dict[str, List[Task]]]) -> List[pattern_type]:
        wrappers = list(map(load_task, next(iter(created[0].values()))))
        return [ModelPattern(init_method=lambda: m) for m in wrappers]

    if params is None:
        params = {
            "optimizer": String(Choice(values=["sgd", "rmsprop", "adam"])),
            "optimizer_config": {
                "lr": Float(Exponential(1e-5, 0.1))
            }
        }

    if metrics is None:
        if task_type is None:
            raise ValueError("either `task_type` or `metrics` should be provided")
        if task_type is TaskTypes.CLASSIFICATION:
            metrics = ["acc", "auc"]
        else:
            metrics = ["mae", "mse"]
    estimators = list(map(Estimator, metrics))

    hpo = HPOBase.make(
        hpo_method, _creator, params,
        converter=_converter, verbose_level=verbose_level
    )
    if hpo.is_sequential:
        if num_jobs is None:
            num_jobs = 0
        if num_jobs > 1:
            print(
                f"{LoggingMixin.warning_prefix}`num_jobs` is set but hpo is sequential, "
                "please use `num_parallel` instead"
            )
        num_jobs = 0
    if search_config is None:
        search_config = {}
    update_dict({
        "num_retry": 1, "num_search": num_search,
        "score_weights": score_weights, "estimator_scoring_function": estimator_scoring_function
    }, search_config)
    if num_jobs is not None:
        search_config["num_jobs"] = num_jobs
    search_config.setdefault("parallel_logging_folder", os.path.join(temp_folder, "__hpo_parallel__"))
    hpo.search(x, y, estimators, x_cv, y_cv, **search_config)
    return hpo


def _to_wrappers(wrappers: wrappers_type) -> wrappers_dict_type:
    if not isinstance(wrappers, dict):
        if not isinstance(wrappers, list):
            wrappers = [wrappers]
        names = [wrapper.model.__identifier__ for wrapper in wrappers]
        if len(set(names)) != len(wrappers):
            raise ValueError("wrapper names are not provided but identical wrapper.model is detected")
        wrappers = dict(zip(names, wrappers))
    return wrappers


def estimate(x: data_type,
             y: data_type = None,
             *,
             wrappers: wrappers_type = None,
             metrics: Union[str, List[str]] = None,
             other_patterns: Dict[str, patterns_type] = None,
             **kwargs) -> Comparer:
    patterns = {}
    if isinstance(metrics, str):
        metrics = [metrics]
    if wrappers is None:
        if y is None:
            raise ValueError("either `wrappers` or `y` should be provided")
        if metrics is None:
            raise ValueError("either `wrappers` or `metrics` should be provided")
        if other_patterns is None:
            raise ValueError("either `wrappers` or `other_patterns` should be provided")
    else:
        wrappers = _to_wrappers(wrappers)
        for name, wrapper in wrappers.items():
            if y is None:
                x, y = wrapper.tr_data.read_file(x)
                y = wrapper.tr_data.transform(x, y).y
            if metrics is None:
                metrics = [k for k, v in wrapper.pipeline.metrics.items() if v is not None]
            with eval_context(wrapper.model):
                patterns[name] = ModelPattern(
                    predict_method=partial(wrapper.predict, **kwargs),
                    predict_prob_method=partial(wrapper.predict_prob, **kwargs)
                )
    if other_patterns is not None:
        for other_name in other_patterns.keys():
            if other_name in patterns:
                prefix = LoggingMixin.warning_prefix
                print(f"{prefix}'{other_name}' is found in `other_patterns`, it will be overwritten")
        update_dict(other_patterns, patterns)
    estimators = list(map(Estimator, metrics))
    comparer = Comparer(patterns, estimators)
    comparer.compare(x, y)
    return comparer


def save(wrappers: wrappers_type,
         identifier: str = "cflearn",
         saving_folder: str = None) -> wrappers_dict_type:
    wrappers = _to_wrappers(wrappers)
    saving_path = _to_saving_path(identifier, saving_folder)
    for name, wrapper in wrappers.items():
        wrapper.save(_make_saving_path(name, saving_path, True), compress=True)
    return wrappers


def load(identifier: str = "cflearn",
         saving_folder: str = None) -> wrappers_dict_type:
    wrappers = {}
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
            wrappers[name] = Wrapper.load(_make_saving_path(name, saving_path, False), compress=True)
    if not wrappers:
        raise ValueError(f"'{saving_path}' was not a valid saving path")
    return wrappers


# others

def make_toy_model(config: Dict[str, Any] = None,
                   *,
                   model: str = "fcnn",
                   task_type: str = "reg",
                   data_tuple: Tuple[List] = None) -> Wrapper:
    if config is None:
        config = {}
    if data_tuple is None:
        if task_type == "reg":
            data_tuple = [[0]], [[1]]
        else:
            data_tuple = [[0], [1]], [[1], [0]]
    base_config = {
        "model": model,
        "model_config": {"mapping_configs": {"batch_norm": False}},
        "cv_split": 0.,
        "trigger_logging": False,
        "data_config": {"valid_columns": [0], "task_type": TaskTypes.from_str(task_type)}
    }
    wrapper = Wrapper(update_dict(config, base_config), verbose_level=0)
    return wrapper.fit(*data_tuple)


__all__ = [
    "ModelBase", "Pipeline", "Wrapper",
    "register_metric", "register_optimizer", "register_scheduler",
    "make", "save", "load", "estimate", "ensemble", "repeat_with", "tune_with", "make_toy_model",
    "Initializer", "register_initializer",
    "Processor", "register_processor"
]
