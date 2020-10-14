import os
import optuna
import shutil

import numpy as np
import cftool.ml.param_utils as pu

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cfdata.tabular import *
from optuna.trial import Trial
from cftool.ml.hpo import HPOBase
from cfdata.tabular.misc import split_file

from .basic import *
from ..dist import *
from ..misc.toolkit import *
from .ensemble import *
from ..bases import Wrapper


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

        self.has_column_names = None
        if y is not None:
            y, y_cv = map(to_2d, [y, y_cv])
            if need_cv_split:
                data = TabularData.simple(task_type).read(x, y)
                split = data.split(hpo_cv_split, order=hpo_cv_split_order)
                tr_data, cv_data = split.remained, split.split
                x, y = tr_data.raw.xy
                x_cv, y_cv = cv_data.raw.xy
        elif isinstance(x, str):
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
            self.has_column_names = tr_data._has_column_names

            if not need_cv_split:
                y = tr_data.processed.y
                if x_cv is not None:
                    if y_cv is None:
                        y_cv = tr_data.transform(x_cv).y
                    else:
                        y_cv = tr_data.transform_labels(y_cv)
            else:
                split = tr_data.split(hpo_cv_split, order=hpo_cv_split_order)
                tr_data, cv_data = split.remained, split.split
                y, y_cv = tr_data.processed.y, cv_data.processed.y
                indices_pair = (split.remained_indices, split.split_indices)
                x, x_cv = split_file(
                    x,
                    export_folder="_split",
                    indices_pair=indices_pair,
                )
                x, x_cv = map(os.path.abspath, [x, x_cv])
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
        *,
        sequential: bool = False,
    ) -> Union[List[Union[Task, Wrapper]], Wrapper]:
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
        if num_repeat <= 1 or sequential:
            get = lambda params_: make(model, **params_).fit(x, y, x_cv, y_cv)
            if num_repeat <= 1:
                params["logging_folder"] = temp_folder
                return get(params)
            wrappers = []
            for i in range(num_repeat):
                local_params = shallow_copy_dict(params)
                local_params["logging_folder"] = os.path.join(temp_folder, str(i))
                wrappers.append(get(local_params))
            return wrappers
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


def _init_extra_config(
    metrics: Union[str, List[str]] = None,
    score_weights: Union[Dict[str, float], None] = None,
    extra_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    new = extra_config or {}
    new.setdefault("use_timing_context", False)
    if metrics is not None:
        metric_config = new.setdefault("metric_config", {})
        metric_config["types"] = metrics
        if score_weights is not None:
            metric_config["weights"] = score_weights
    return new


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

    extra_config = _init_extra_config(metrics, score_weights, extra_config)
    tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    x, y, x_cv, y_cv = tuner.x, tuner.y, tuner.x_cv, tuner.y_cv

    created_type = Union[Dict[str, List[Task]], ModelPattern]

    def _creator(_, __, params_) -> created_type:
        num_jobs_ = num_parallel if hpo.is_sequential else 0
        temp_folder_ = temp_folder
        if num_repeat <= 1:
            temp_folder_ = os.path.join(temp_folder, hash_code(str(params_)))
        results = tuner.train(model, params_, num_repeat, num_jobs_, temp_folder_)
        if num_repeat <= 1:
            return results.to_pattern(contains_labels=True)
        return {model: results}

    if num_repeat <= 1:
        _converter = None
    else:

        def _converter(created: List[created_type]) -> List[pattern_type]:
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
        hpo_method,
        _creator,
        params,
        converter=_converter,
        verbose_level=verbose_level,
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

    def pop(self, trial: Trial) -> Any:
        method = getattr(trial, f"suggest_{self.dtype}")
        if self.dtype == "categorical":
            return method(self.name, self.values)
        low, high = self.values
        config = {} if self.config is None else self.config
        return method(self.name, low, high, **config)


optuna_params_type = Dict[str, Union[Union[OptunaParam, Any], "optuna_params_type"]]


class OptunaParamConverter:
    prefix = "[^optuna^]"

    def get_usage(self, k: str) -> Union[str, None]:
        if not k.startswith(self.prefix):
            return
        usage_k = k[len(self.prefix) :]
        if not usage_k.startswith("[") or not usage_k.endswith("]"):
            msg = f"special keys must end with '[]' to indicate its usage"
            raise ValueError(msg)
        return usage_k[1:-1]

    def convert(self, optuna_params: optuna_params_type) -> optuna_params_type:
        def _inner(d: optuna_params_type, current: dict):
            for k, v in d.items():
                usage = self.get_usage(k)
                if usage is not None:
                    attr = getattr(self, f"_convert_{usage}", None)
                    if attr is None:
                        raise NotImplementedError(f"unrecognized usage '{usage}' found")
                    current[k] = attr(v)
                    continue
                if isinstance(v, dict):
                    _inner(v, current.setdefault(k, {}))
                    continue
                if isinstance(v, OptunaParam):
                    current[k] = v
                    continue

        new = {}
        _inner(optuna_params, new)
        return new

    @staticmethod
    def _convert_hidden_units(value: Any) -> optuna_params_type:
        # parse
        config = {}
        prefix, low, high, num_layers, *args = value.split("_")
        low, high, num_layers = map(int, [low, high, num_layers])
        if args:
            if len(args) == 1:
                if args[0] == "log":
                    config["log"] = True
                else:
                    config["step"] = int(args[0])
            else:
                config["log"] = True
                config["step"] = int(args[0])
        # inject
        num_layers_name = f"{prefix}_num_layers"
        rs = {"num_layers": OptunaParam(num_layers_name, [1, num_layers], "int")}
        for i in range(num_layers):
            key = f"hidden_unit_{i}"
            rs[key] = OptunaParam(f"{prefix}_{key}", [low, high], "int", config)
        return rs

    @staticmethod
    def _convert_dndf_config(value: Any) -> optuna_params_type:
        prefix, num_tree, tree_depth = value.split("_")
        num_tree, tree_depth = map(int, [num_tree, tree_depth])
        num_tree = max(4, num_tree)
        tree_depth = max(2, tree_depth)
        use_dndf_key = f"{prefix}_use_dndf"
        num_tree_key = f"{prefix}_num_tree"
        tree_depth_key = f"{prefix}_tree_depth"
        return {
            "use_dndf": OptunaParam(use_dndf_key, [True, False], "categorical"),
            "num_tree": OptunaParam(num_tree_key, [4, num_tree], "int", {"log": True}),
            "tree_depth": OptunaParam(tree_depth_key, [2, tree_depth], "int"),
        }

    @staticmethod
    def _convert_pruner_config(value: Any) -> optuna_params_type:
        prefix = value
        available_methods = ["auto_prune", "surgery", "simplified"]
        use_pruner_key = f"{prefix}_use_pruner"
        method_key = f"{prefix}_method"
        return {
            "use_pruner": OptunaParam(use_pruner_key, [True, False], "categorical"),
            "method": OptunaParam(method_key, available_methods, "categorical"),
        }

    def pop(
        self,
        usage: str,
        value: Dict[str, OptunaParam],
        trial: Trial,
    ) -> Any:
        attr = getattr(self, f"_parse_{usage}", None)
        if attr is None:
            raise NotImplementedError(f"unrecognized usage '{usage}' found")
        return attr(value, trial)

    def parse(self, usage: str, value: Dict[str, Any]) -> Any:
        attr = getattr(self, f"_parse_{usage}", None)
        if attr is None:
            raise NotImplementedError(f"unrecognized usage '{usage}' found")
        return attr(value, None)

    @staticmethod
    def _parse_hidden_units(d: Dict[str, Any], trial: Trial) -> Any:
        hidden_units = []
        num_layers = d["num_layers"]
        if trial is not None:
            max_layers = num_layers.values[1]
            num_layers = num_layers.pop(trial)
            for i in range(num_layers, max_layers):
                d.pop(f"hidden_unit_{i}", None)
        for i in range(num_layers):
            key = f"hidden_unit_{i}"
            hidden_unit = d.pop(key, None)
            if hidden_unit is None:
                raise ValueError(f"'{key}' is not found in `model_config`")
            if trial is not None:
                hidden_unit = hidden_unit.pop(trial)
            hidden_units.append(hidden_unit)
        return hidden_units

    @staticmethod
    def _parse_dndf_config(d: Dict[str, Any], trial: Trial) -> Any:
        use_dndf = d["use_dndf"]
        if trial is not None:
            use_dndf = use_dndf.pop(trial)
        if not use_dndf:
            return
        num_tree, tree_depth = d["num_tree"], d["tree_depth"]
        if trial is not None:
            num_tree = num_tree.pop(trial)
            tree_depth = tree_depth.pop(trial)
        return {"num_tree": num_tree, "tree_depth": tree_depth}

    @staticmethod
    def _parse_pruner_config(d: Dict[str, Any], trial: Trial) -> Any:
        use_pruner = d["use_pruner"]
        if trial is not None:
            use_pruner = use_pruner.pop(trial)
        if not use_pruner:
            return
        method = d["method"]
        if trial is not None:
            method = method.pop(trial)
        return {"method": method}

    # api

    @classmethod
    def make_hidden_units(
        cls,
        prefix: str,
        low: int,
        high: int,
        num_layers: int,
        step: int = 1,
        log: bool = True,
    ) -> Dict[str, str]:
        key = f"{cls.prefix}[hidden_units]"
        value = f"{prefix}_{low}_{high}_{num_layers}_{step}"
        if log:
            value = f"{value}_log"
        return {key: value}

    @classmethod
    def make_dndf_config(
        cls,
        prefix: str,
        num_tree: int,
        tree_depth: int,
    ) -> Dict[str, str]:
        key = f"{cls.prefix}[dndf_config]"
        value = f"{prefix}_{num_tree}_{tree_depth}"
        return {key: value}

    @classmethod
    def make_pruner_config(
        cls,
        prefix: str,
    ) -> Dict[str, str]:
        key = f"{cls.prefix}[pruner_config]"
        value = prefix
        return {key: value}


class OptunaKeyMapping:
    def __init__(self, tuner: _Tuner, optuna_params: optuna_params_type):
        self.tuner = tuner
        self.delim = SAVING_DELIM
        self.converter = OptunaParamConverter()
        self.params = self.converter.convert(optuna_params)
        self.optuna_key_mapping: Dict[str, str] = {}

        def _inject_mapping(d: optuna_params_type, prefix_list: List[str]):
            for k, v in d.items():
                new_prefix_list = prefix_list + [k]
                if isinstance(v, OptunaParam):
                    self.optuna_key_mapping[v.name] = self.delim.join(new_prefix_list)
                    continue
                _inject_mapping(v, new_prefix_list)

        _inject_mapping(self.params, [])

    def pop(self, trial: Trial) -> Dict[str, Any]:
        optuna_params = shallow_copy_dict(self.params)
        current_params = shallow_copy_dict(self.tuner.base_params)

        def _inject_suggestion(d: optuna_params_type, current: dict):
            for k, v in d.items():
                usage = self.converter.get_usage(k)
                if usage is not None:
                    current[usage] = self.converter.pop(usage, v, trial)
                    continue
                if isinstance(v, dict):
                    _inject_suggestion(v, current.setdefault(k, {}))
                    continue
                current[k] = v.pop(trial)

        _inject_suggestion(optuna_params, current_params)
        return current_params

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

    def convert(self, param_values: Dict[str, Any]) -> Dict[str, Any]:
        converted = {}

        def _inject_values(d: Dict[str, Any], current: dict):
            for k, v in d.items():
                usage = self.converter.get_usage(k)
                if usage is not None:
                    current[usage] = self.converter.parse(usage, v)
                    continue
                if isinstance(v, dict):
                    _inject_values(v, current.setdefault(k, {}))
                    continue
                current[k] = v

        _inject_values(param_values, converted)
        return converted


class OptunaResult(NamedTuple):
    tuner: _Tuner
    study: optuna.study.Study
    optuna_key_mapping: OptunaKeyMapping

    @property
    def best_param(self) -> Dict[str, Any]:
        param = shallow_copy_dict(self.tuner.base_params)
        optuna_param = self.optuna_key_mapping.parse(self.study.best_params)
        update_dict(optuna_param, param)
        return self.optuna_key_mapping.convert(param)


class OptunaPresetParams:
    def __init__(self):
        lr_param = OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
        optim_param = OptunaParam(
            "optimizer", ["nag", "rmsprop", "adam", "adamw"], "categorical"
        )
        default_init_param = OptunaParam(
            "default_init_method", [None, "truncated_normal"], "categorical"
        )
        self.base_params = {
            "optimizer": optim_param,
            "optimizer_config": {"lr": lr_param},
            "model_config": {
                "default_encoding_configs": {"init_method": default_init_param},
            },
        }

    def get(self, model: str) -> optuna_params_type:
        attr = getattr(self, f"_{model}_preset", None)
        if attr is None:
            raise NotImplementedError(f"preset params for '{model}' is not defined")
        return attr()

    def _fcnn_preset(self) -> optuna_params_type:
        params = shallow_copy_dict(self.base_params)
        model_config = params["model_config"]
        model_config.update(OptunaParamConverter.make_hidden_units("mlp", 8, 2048, 3))
        mapping_config = {
            "dropout": OptunaParam("mlp_dropout", [0.0, 0.9], "float"),
            "batch_norm": OptunaParam("mlp_batch_norm", [False, True], "categorical"),
        }
        mapping_config.update(OptunaParamConverter.make_pruner_config("mlp"))
        model_config["mapping_configs"] = mapping_config
        model_config["default_encoding_configs"]["embedding_dim"] = OptunaParam(
            "embedding_dim", [8, "auto"], "categorical"
        )
        return params

    def _tree_dnn_preset(self) -> optuna_params_type:
        params = self._fcnn_preset()
        model_config = params["model_config"]
        model_config.update(OptunaParamConverter.make_dndf_config("dndf", 128, 8))
        return params


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
    num_jobs: int = 4,
    num_trial: int = 50,
    num_repeat: int = 5,
    num_parallel: int = 0,
    timeout: float = None,
    score_weights: Union[Dict[str, float], None] = None,
    estimator_scoring_function: Union[str, scoring_fn_type] = "default",
    temp_folder: str = "__tmp__",
    extra_config: Dict[str, Any] = None,
) -> OptunaResult:
    if params is None:
        params = OptunaPresetParams().get(model)

    extra_config = _init_extra_config(metrics, score_weights, extra_config)
    tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    key_mapping = OptunaKeyMapping(tuner, params)

    def objective(trial: Trial) -> float:
        temp_folder_ = os.path.join(temp_folder, str(trial.number))
        current_params = key_mapping.pop(trial)
        current_params["trial"] = trial
        args = model, current_params, num_repeat, num_parallel, temp_folder_
        if num_repeat <= 1:
            m = tuner.train(*args)
            comparer = estimate(
                tuner.x_cv,
                tuner.y_cv,
                wrappers=m,
                metrics=metrics,
                contains_labels=True,
                comparer_verbose_level=6,
            )
        else:
            estimators = tuner.make_estimators(metrics)
            wrappers = tuner.train(*args, sequential=True)
            patterns = [m.to_pattern(contains_labels=True) for m in wrappers]
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

    return OptunaResult(tuner, study, key_mapping)


class Opt:
    def __init__(self, task_type: TaskTypes):
        self.task_type = task_type

    @property
    def study(self) -> optuna.study.Study:
        return self.optuna_result.study

    def _merge_data(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
    ) -> Tuple[data_type, data_type]:
        if x_cv is None:
            return x, y
        if not isinstance(x_cv, str):
            if isinstance(x_cv, list):
                x_merged = x + x_cv
                y_merged = None if y is None else y + y_cv
            else:
                x_merged = np.vstack([x, x_cv])
                y_merged = None if y is None else np.vstack([y, y_cv])
            return x_merged, y_merged
        has_column_names = self.optuna_result.tuner.has_column_names
        with open(x, "r") as fx, open(x_cv, "r") as fx_cv:
            x_lines = fx.readlines()
            x_cv_lines = fx_cv.readlines()
            if has_column_names:
                x_cv_lines = x_cv_lines[1:]
        x_merged_lines = x_lines + x_cv_lines
        path, ext = os.path.splitext(x)
        new_file = f"{path}_^merged^{ext}"
        with open(new_file, "w") as f:
            f.write("".join(x_merged_lines))

    def optimize(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        model: str = "fcnn",
        study_config: Dict[str, Any] = None,
        metrics: Union[str, List[str]] = None,
        num_jobs: int = 4,
        num_trial: int = 50,
        num_repeat: int = 5,
        num_parallel: int = 0,
        timeout: float = None,
        score_weights: Union[Dict[str, float], None] = None,
        estimator_scoring_function: Union[str, scoring_fn_type] = "default",
        temp_folder: str = "__tmp__",
        extra_config: Dict[str, Any] = None,
        num_final_repeat: int = 10,
        bagging_config: Dict[str, Any] = None,
    ) -> "Opt":
        optuna_temp_folder = os.path.join(temp_folder, "__optuna__")
        self.optuna_result = optuna_tune(
            x,
            y,
            x_cv,
            y_cv,
            model=model,
            task_type=self.task_type,
            study_config=study_config,
            metrics=metrics,
            num_jobs=num_jobs,
            num_trial=num_trial,
            num_repeat=num_repeat,
            num_parallel=num_parallel,
            timeout=timeout,
            score_weights=score_weights,
            estimator_scoring_function=estimator_scoring_function,
            temp_folder=optuna_temp_folder,
            extra_config=extra_config,
        )
        self.best_param = self.optuna_result.best_param
        if bagging_config is not None:
            self.repeat_result = None
            bagging_temp_folder = os.path.join(temp_folder, "__bagging__")
            bagging = Ensemble(self.task_type, self.best_param).bagging
            bagging_config.setdefault("task_name", f"{model}_opt")
            increment_config = bagging_config.setdefault("increment_config", {})
            increment_config.setdefault("trigger_logging", False)
            increment_config.setdefault("verbose_level", 0)
            bagging_config["temp_folder"] = bagging_temp_folder
            bagging_config["use_tracker"] = False
            bagging_config["num_jobs"] = max(num_parallel, num_jobs)
            bagging_config["models"] = model
            bagging_config["k"] = num_final_repeat
            x_merged, y_merged = self._merge_data(x, y, x_cv, y_cv)
            self.bagging_result = bagging(x_merged, y_merged, **bagging_config)
            self.pattern = self.bagging_result.pattern
            self.data = self.bagging_result.data
        else:
            self.bagging_result = None
            repeat_temp_folder = os.path.join(temp_folder, "__repeat__")
            self.repeat_result = repeat_with(
                x,
                y,
                x_cv,
                y_cv,
                **self.best_param,
                models=model,
                temp_folder=repeat_temp_folder,
                num_repeat=num_final_repeat,
                num_jobs=num_jobs,
            )
            self.pattern = ensemble(self.repeat_result.patterns[model])
            self.data = self.repeat_result.data
        return self


__all__ = [
    "tune_with",
    "optuna_tune",
    "OptunaParam",
    "OptunaParamConverter",
    "OptunaPresetParams",
    "Opt",
]
