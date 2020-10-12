import os
import optuna
import shutil
import cftool.ml.param_utils as pu

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cfdata.tabular import *
from cftool.ml.hpo import HPOBase
from cfdata.tabular.misc import split_file

from .basic import *
from ..dist import *
from ..misc.toolkit import *


class _Tuner:
    def __init__(
        self,
        x: data_type,
        y: data_type,
        x_cv: data_type,
        y_cv: data_type,
        task_type: TaskTypes,
        **kwargs,
    ):
        hpo_cv_split = kwargs.get("hpo_cv_split", 0.1)
        hpo_cv_split_order = kwargs.get("hpo_cv_split_order", "auto")
        need_cv_split = x_cv is None and hpo_cv_split > 0.0

        if y is not None:
            y, y_cv = map(to_2d, [y, y_cv])
            if need_cv_split:
                data = TabularData.simple(task_type).read(x, y)
                split = data.split(hpo_cv_split, order=hpo_cv_split_order)
                tr_data, cv_data = split.remained, split.split
                x, y = tr_data.raw.xy
                x_cv, y_cv = cv_data.raw.xy
        elif isinstance(x, str):
            if need_cv_split:
                print(
                    f"{LoggingMixin.warning_prefix}only random split is supported "
                    f"for file datasets, `split_order` ({hpo_cv_split_order}) "
                    "will be ignored"
                )
                x_cv, x = split_file(x, export_folder="_split", split=hpo_cv_split)
                x, x_cv = map(os.path.abspath, [x, x_cv])
            data_config = kwargs.get("data_config", {})
            data_config["task_type"] = task_type
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
            tr_data = TabularData(**data_config)
            tr_data.read(x, **read_config)
            y = tr_data.processed.y
            if x_cv is not None:
                if y_cv is None:
                    y_cv = tr_data.transform(x_cv).y
                else:
                    y_cv = tr_data.transform_labels(y_cv)
        else:
            raise ValueError("`x` should be a file when `y` is not provided")

        self.task_type = task_type
        self.x, self.x_cv = x, x_cv
        self.y, self.y_cv = y, y_cv
        self.base_params = shallow_copy_dict(kwargs)

    def make_estimators(self, metrics: Union[str, List[str]]) -> List[Estimator]:
        if metrics is None:
            if self.task_type is None:
                raise ValueError("either `task_type` or `metrics` should be provided")
            if self.task_type is TaskTypes.CLASSIFICATION:
                metrics = ["acc", "auc"]
            else:
                metrics = ["mae", "mse"]
        return list(map(Estimator, metrics))

    def train(
        self,
        model: str,
        params: Dict[str, Any],
        num_repeat: int,
        num_parallel: int,
        temp_folder: str,
    ) -> List[Task]:
        identifier = hash_code(str(params))
        params = update_dict(params, shallow_copy_dict(self.base_params))
        params["verbose_level"] = 0
        params["use_tqdm"] = False
        if isinstance(self.x, str):
            y = y_cv = None
            x, x_cv = self.x, self.x_cv
        else:
            x, x_cv = self.x.copy(), self.x_cv.copy()
            y = self.y.copy()
            y_cv = None if self.y_cv is None else self.y_cv.copy()
        results = repeat_with(
            x,
            y,
            x_cv,
            y_cv,
            num_repeat=num_repeat,
            num_jobs=num_parallel,
            models=model,
            identifiers=identifier,
            temp_folder=temp_folder,
            return_tasks=True,
            return_patterns=False,
            **params,
        )
        return results.experiments.tasks[identifier]


class HPOResult(NamedTuple):
    hpo: HPOBase
    extra_config: Dict[str, Any]

    @property
    def best_param(self) -> Dict[str, Any]:
        param = shallow_copy_dict(self.hpo.best_param)
        return update_dict(param, shallow_copy_dict(self.extra_config))


def tune_with(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    hpo_method: str = "bo",
    params: pu.params_type = None,
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
    extra_config: Dict[str, Any] = None,
) -> HPOResult:

    if os.path.isdir(temp_folder):
        print(
            f"{LoggingMixin.warning_prefix}'{temp_folder}' already exists, it will be overwritten"
        )
        shutil.rmtree(temp_folder)

    if extra_config is None:
        extra_config = {}
    tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    x, y, x_cv, y_cv = tuner.x, tuner.y, tuner.x_cv, tuner.y_cv

    def _creator(_, __, params_) -> Dict[str, List[Task]]:
        num_jobs_ = num_parallel if hpo.is_sequential else 0
        tasks = tuner.train(model, params_, num_repeat, num_jobs_, temp_folder)
        return {model: tasks}

    def _converter(created: List[Dict[str, List[Task]]]) -> List[pattern_type]:
        return tasks_to_patterns(created[0][model], contains_labels=True)

    if params is None:
        default_init_param = pu.Any(pu.Choice(values=[None, "truncated_normal"]))
        params = {
            "optimizer": pu.String(pu.Choice(values=["sgd", "rmsprop", "adam"])),
            "optimizer_config": {"lr": pu.Float(pu.Exponential(1e-5, 0.1))},
            "model_config": {
                "default_encoding_configs": {"init_method": default_init_param},
            },
        }

    hpo = HPOBase.make(
        hpo_method, _creator, params, converter=_converter, verbose_level=verbose_level
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
    update_dict(
        {
            "num_retry": 1,
            "num_search": num_search,
            "score_weights": score_weights,
            "estimator_scoring_function": estimator_scoring_function,
        },
        search_config,
    )
    if num_jobs is not None:
        search_config["num_jobs"] = num_jobs
    search_config.setdefault(
        "parallel_logging_folder", os.path.join(temp_folder, "__hpo_parallel__")
    )
    estimators = tuner.make_estimators(metrics)
    hpo.search(x, y, estimators, x_cv, y_cv, **search_config)
    return HPOResult(hpo, extra_config)


class OptunaParam(NamedTuple):
    name: str
    values: Any
    dtype: str  # [int | float | categorical]
    config: Dict[str, Any] = None

    def pop(self, trial: optuna.trial.Trial) -> Any:
        method = getattr(trial, f"suggest_{self.dtype}")
        if self.dtype == "categorical":
            return method(self.name, self.values)
        low, high = self.values
        config = {} if self.config is None else self.config
        return method(self.name, low, high, **config)


optuna_params_type = Dict[str, Union[OptunaParam, "optuna_params_type"]]


class OptunaKeyMapping:
    def __init__(self, optuna_params: optuna_params_type):
        self.delim = SAVING_DELIM
        self.params = optuna_params
        self.optuna_key_mapping: Dict[str, str] = {}

        def _inject_mapping(d: optuna_params_type, prefix_list: List[str]):
            for k, v in d.items():
                new_prefix_list = prefix_list + [k]
                if isinstance(v, OptunaParam):
                    self.optuna_key_mapping[v.name] = self.delim.join(new_prefix_list)
                    continue
                _inject_mapping(v, new_prefix_list)

        _inject_mapping(optuna_params, [])

    def parse(self, optuna_param_values: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        for k, v in optuna_param_values.items():
            key_mapping = self.optuna_key_mapping[k]
            key_path = key_mapping.split(self.delim)
            local_param = params
            for sub_k in key_path[:-1]:
                local_param = local_param.setdefault(sub_k, {})
            local_param[key_path[-1]] = v
        return params


class OptunaResult(NamedTuple):
    tuner: _Tuner
    study: optuna.study.Study
    optuna_key_mapping: OptunaKeyMapping
    extra_config: Dict[str, Any]

    @property
    def best_param(self) -> Dict[str, Any]:
        param = shallow_copy_dict(self.tuner.base_params)
        optuna_param = self.optuna_key_mapping.parse(self.study.best_params)
        update_dict(optuna_param, param)
        self.get_hidden_units(param, None)
        return update_dict(param, shallow_copy_dict(self.extra_config))

    @staticmethod
    def get_hidden_units(
        params: Dict[str, Any],
        trial: Union[optuna.trial.Trial, None],
    ) -> Union[List[int], None]:
        hidden_units = None
        model_config = params.setdefault("model_config", {})
        num_layers = model_config.pop("num_layers", None)
        if num_layers is not None:
            hidden_units = []
            if trial is not None:
                max_layers = num_layers.values[1]
                num_layers = num_layers.pop(trial)
                for i in range(num_layers, max_layers):
                    model_config.pop(f"hidden_unit_{i}", None)
            for i in range(num_layers):
                key = f"hidden_unit_{i}"
                hidden_unit = model_config.pop(key, None)
                if hidden_unit is None:
                    raise ValueError(f"'{key}' is not found in `model_config`")
                if trial is not None:
                    hidden_unit = hidden_unit.pop(trial)
                hidden_units.append(hidden_unit)
            if trial is None:
                model_config["hidden_units"] = hidden_units
        return hidden_units


def optuna_tune(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    task_type: TaskTypes = None,
    params: optuna_params_type = None,
    study_config: Dict[str, Any] = None,
    metrics: Union[str, List[str]] = None,
    num_jobs: int = 1,
    num_trial: int = 10,
    num_repeat: int = 5,
    num_parallel: int = 4,
    timeout: float = None,
    score_weights: Union[Dict[str, float], None] = None,
    estimator_scoring_function: Union[str, scoring_fn_type] = "default",
    temp_folder: str = "__tmp__",
    extra_config: Dict[str, Any] = None,
) -> OptunaResult:
    if params is None:
        lr_param = OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
        optim_param = OptunaParam(
            "optimizer", ["sgd", "rmsprop", "adam"], "categorical"
        )
        default_init_param = OptunaParam(
            "default_init_method", [None, "truncated_normal"], "categorical"
        )
        params = {
            "optimizer": optim_param,
            "optimizer_config": {"lr": lr_param},
            "model_config": {
                "default_encoding_configs": {"init_method": default_init_param},
            },
        }
    key_mapping = OptunaKeyMapping(params)

    if extra_config is None:
        extra_config = {}
    tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    estimators = tuner.make_estimators(metrics)

    def objective(trial: optuna.trial.Trial) -> float:
        optuna_params = shallow_copy_dict(params)
        current_params = shallow_copy_dict(tuner.base_params)
        # handle hidden units
        model_config = current_params.setdefault("model_config", {})
        hidden_units = OptunaResult.get_hidden_units(optuna_params, trial)
        if hidden_units is not None:
            model_config["hidden_units"] = hidden_units
        # get other suggestions
        def _inject_suggestion(d: optuna_params_type, current: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    _inject_suggestion(v, current.setdefault(k, {}))
                    continue
                current[k] = v.pop(trial)

        _inject_suggestion(optuna_params, current_params)
        # train & estimate
        args = model, current_params, num_repeat, num_parallel, temp_folder
        patterns = tasks_to_patterns(tuner.train(*args), contains_labels=True)
        comparer = Comparer({model: patterns}, estimators)
        comparer.compare(
            tuner.x_cv,
            tuner.y_cv,
            scoring_function=estimator_scoring_function,
            verbose_level=6,
        )
        scores = {k: v[model] for k, v in comparer.final_scores.items()}
        if score_weights is None:
            score = sum(scores.values()) / len(scores)
        else:
            weighted = sum(score * score_weights[k] for k, score in scores.items())
            score = weighted / sum(score_weights.values())
        return score

    if study_config is None:
        study_config = {}
    study_config["direction"] = "maximize"
    study_config.setdefault("study_name", f"{model}_optuna")
    study = optuna.create_study(**study_config)
    study.optimize(objective, num_trial, timeout, num_jobs)

    return OptunaResult(tuner, study, key_mapping, extra_config)


__all__ = [
    "tune_with",
    "optuna_tune",
    "OptunaParam",
]
