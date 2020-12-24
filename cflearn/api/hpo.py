import os
import torch
import optuna
import shutil

import numpy as np
import cftool.ml.param_utils as pu

from typing import *
from cftool.misc import hash_code
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cftool.ml.hpo import HPOBase
from cftool.ml.utils import pattern_type
from cftool.ml.utils import scoring_fn_type
from cftool.ml.utils import Metrics
from cftool.ml.utils import Estimator
from cfdata.types import np_float_type
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import TaskTypes
from optuna.trial import Trial

from .basic import *
from ..misc.toolkit import *
from .ensemble import Ensemble
from ..types import data_type
from ..pipeline import Pipeline
from ..protocol import DataProtocol
from ..misc._api import SAVING_DELIM
from ..misc.toolkit import inject_mlflow_stuffs


class _TunerResult(NamedTuple):
    model: str
    repeat_result: RepeatResult

    @property
    def pipelines(self) -> List[Pipeline]:
        pipelines = self.repeat_result.pipelines
        if pipelines is None:
            raise ValueError("`pipelines` are not yet generated")
        return pipelines[self.model]

    @property
    def patterns(self) -> List[ModelPattern]:
        patterns = self.repeat_result.patterns
        if patterns is None:
            raise ValueError("`patterns` are not yet generated")
        return patterns[self.model]

    @property
    def weighted_metrics(self) -> Dict[str, np.ndarray]:
        weighted_metrics = {}
        final_results_dict = self.repeat_result.final_results
        if final_results_dict is None:
            raise ValueError("training process is corrupted")
        final_results = final_results_dict[self.model]
        for key in sorted(final_results[0].weighted_scores.keys()):
            local_metrics = [result.weighted_metrics[key] for result in final_results]
            weighted_metrics[key] = np.array(local_metrics, np_float_type)
        return weighted_metrics


class _Tuner(LoggingMixin):
    data_file = "__data__.pt"
    config_name = "__config__"

    def __init__(
        self,
        x: data_type = None,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        task_type: task_type_type = "",
        **kwargs: Any,
    ):
        # `x` will be None if `load` is called
        if x is None:
            return

        kwargs = shallow_copy_dict(kwargs)
        self.has_column_names = None
        if y is not None:
            y, y_cv = map(to_2d, [y, y_cv])
        elif isinstance(x, str):
            data_config = kwargs.get("data_config", {})
            data_config["task_type"] = task_type
            read_config = kwargs.get("read_config", {})
            delim = read_config.get("delim", kwargs.get("delim"))
            if delim is not None:
                read_config["delim"] = delim
            else:
                print(
                    f"{LoggingMixin.warning_prefix}delimiter of the given "
                    "file dataset is not provided, this may cause incorrect parsing"
                )
            if y is not None:
                read_config["y"] = y
            data_protocol = kwargs.get("data_protocol", "tabular")
            tr_data = DataProtocol.make(data_protocol, **data_config)
            tr_data.read(x, **read_config)
            self.has_column_names = tr_data._has_column_names
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

    def make_estimators(
        self,
        metrics: Optional[Union[str, List[str]]],
        repeat_result: Optional[RepeatResult],
    ) -> List[Estimator]:
        if isinstance(metrics, str):
            metrics = [metrics]
        elif metrics is None:
            if self.task_type is None:
                raise ValueError("either `task_type` or `metrics` should be provided")
            if repeat_result is not None:
                if repeat_result.pipelines is None:
                    raise ValueError("pipelines are not provided in `repeat_result`")
                pipeline = list(repeat_result.pipelines.values())[0][0]
                metrics = []
                for metric in sorted(pipeline.trainer.metrics):
                    if metric in Metrics.sign_dict:
                        metrics.append(metric)
                if not metrics:
                    metrics = None
            if metrics is None:
                if parse_task_type(self.task_type) is TaskTypes.CLASSIFICATION:
                    metrics = ["acc", "auc"]
                else:
                    metrics = ["mae", "mse"]
        return list(map(Estimator, metrics))

    def make_data(self) -> Tuple[data_type, data_type, data_type, data_type]:
        x: data_type
        y: data_type
        if isinstance(self.x, str):
            y = y_cv = None
            x, x_cv = self.x, self.x_cv
        else:
            assert isinstance(self.x, (list, np.ndarray))
            assert isinstance(self.y, (list, np.ndarray))
            x, y = self.x.copy(), self.y.copy()
            if self.x_cv is None:
                x_cv = None
            else:
                assert isinstance(self.x_cv, (list, np.ndarray))
                x_cv = self.x_cv.copy()
            if self.y_cv is None:
                y_cv = None
            else:
                assert isinstance(self.y_cv, (list, np.ndarray))
                y_cv = self.y_cv.copy()
        return x, y, x_cv, y_cv

    def train(
        self,
        model: str,
        params: Dict[str, Any],
        num_repeat: int,
        num_parallel: int,
        temp_folder: str,
        *,
        compress: bool = True,
        cuda: Optional[str] = None,
        sequential: Optional[bool] = None,
    ) -> _TunerResult:
        params = update_dict(params, shallow_copy_dict(self.base_params))
        params["verbose_level"] = 0
        params["use_tqdm"] = False
        params["cuda"] = cuda
        x, y, x_cv, y_cv = self.make_data()
        repeat_result = repeat_with(
            x,
            y,
            x_cv,
            y_cv,
            num_repeat=num_repeat,
            num_jobs=num_parallel,
            models=model,
            compress=compress,
            temp_folder=temp_folder,
            predict_config={"contains_labels": True},
            sequential=sequential,
            **params,
        )
        return _TunerResult(model, repeat_result)

    def save(self, export_folder: str) -> "_Tuner":
        Saving.prepare_folder(self, export_folder)

        data_path = os.path.join(export_folder, self.data_file)
        data = {"x": self.x, "y": self.y, "x_cv": self.x_cv, "y_cv": self.y_cv}
        torch.save(data, data_path)

        config = {"task_type": self.task_type, "base_params": self.base_params}
        Saving.save_dict(config, self.config_name, export_folder)

        return self

    @classmethod
    def load(cls, export_folder: str) -> "_Tuner":
        instance = cls()
        data_path = os.path.join(export_folder, cls.data_file)
        data = torch.load(data_path)
        instance.x, instance.y = data["x"], data["y"]
        instance.x_cv, instance.y_cv = data["x_cv"], data["y_cv"]
        config = Saving.load_dict(cls.config_name, export_folder)
        instance.task_type = config["task_type"]
        instance.base_params = config["base_params"]
        return instance


class HPOResult(NamedTuple):
    hpo: HPOBase
    extra_config: Dict[str, Any]

    @property
    def best_param(self) -> Dict[str, Any]:
        param = shallow_copy_dict(self.hpo.best_param)
        return update_dict(param, shallow_copy_dict(self.extra_config))


def _init_extra_config(
    metrics: Optional[Union[str, List[str]]] = None,
    score_weights: Optional[Dict[str, float]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    new = extra_config or {}
    new.setdefault("use_timing_context", False)
    if metrics is not None:
        metric_config = new.setdefault("metric_config", {})
        metric_config["types"] = metrics
        if score_weights is not None:
            metric_config["weights"] = score_weights
    return new


default_scoring = "default"


def tune_with(
    x: data_type,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    hpo_method: str = "bo",
    task_type: task_type_type = "",
    params: Optional[pu.params_type] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    num_jobs: Optional[int] = None,
    num_repeat: int = 5,
    num_parallel: int = 1,
    num_search: int = 10,
    temp_folder: str = "__tmp__",
    score_weights: Optional[Dict[str, float]] = None,
    estimator_scoring_function: Union[str, scoring_fn_type] = default_scoring,
    search_config: Optional[Dict[str, Any]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    verbose_level: int = 2,
) -> HPOResult:

    if os.path.isdir(temp_folder):
        print(
            f"{LoggingMixin.warning_prefix}'{temp_folder}' already exists, "
            "it will be overwritten"
        )
        shutil.rmtree(temp_folder)

    extra_config = _init_extra_config(metrics, score_weights, extra_config)
    tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    x, y, x_cv, y_cv = tuner.x, tuner.y, tuner.x_cv, tuner.y_cv

    def _creator(_: Any, __: Any, params_: Dict[str, Any]) -> pattern_type:
        num_jobs_ = num_parallel if hpo.is_sequential else 1
        temp_folder_ = os.path.join(temp_folder, hash_code(str(params_)))
        result = tuner.train(model, params_, num_repeat, num_jobs_, temp_folder_)
        if num_repeat <= 1:
            return result.patterns[0]
        return Ensemble.stacking(result.patterns)

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
        verbose_level=verbose_level,
    )
    if hpo.is_sequential:
        if num_jobs is None:
            num_jobs = 0
        if num_jobs > 1:
            print(
                f"{LoggingMixin.warning_prefix}`num_jobs` is set but hpo is "
                "sequential, please use `num_parallel` instead"
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
        "parallel_logging_folder",
        os.path.join(temp_folder, "__hpo_parallel__"),
    )
    estimators = tuner.make_estimators(metrics, None)
    hpo.search(x, y, estimators, x_cv, y_cv, **search_config)
    return HPOResult(hpo, extra_config)


class OptunaParam(NamedTuple):
    name: str
    values: Any
    dtype: str  # [int | float | categorical]
    config: Optional[Dict[str, Any]] = None

    def pop(self, trial: Trial) -> Any:
        method = getattr(trial, f"suggest_{self.dtype}")
        if self.dtype == "categorical":
            return method(self.name, self.values)
        low, high = self.values
        return method(self.name, low, high, **(self.config or {}))


optuna_params_type = Dict[str, Union[OptunaParam, Dict[str, Any], str]]


class OptunaParamConverter:
    prefix = "[^optuna^]"

    def get_usage(self, k: str) -> Tuple[Optional[str], Optional[str]]:
        if not k.startswith(self.prefix):
            return None, None
        usage_k = k[len(self.prefix) :]
        if not usage_k.startswith("[") or not usage_k.endswith("]"):
            msg = f"special keys must end with '[]' to indicate its usage"
            raise ValueError(msg)
        usage_k = usage_k[1:-1]
        if not usage_k.startswith("[") or "]" not in usage_k:
            return usage_k, None
        user_prefix_end = usage_k.index("]")
        user_prefix = usage_k[1:user_prefix_end]
        usage_k = usage_k[user_prefix_end + 1 :]
        return usage_k, user_prefix

    def convert(self, optuna_params: optuna_params_type) -> optuna_params_type:
        def _inner(d: optuna_params_type, current: dict) -> None:
            for k, v in d.items():
                usage, user_prefix = self.get_usage(k)
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

        new: optuna_params_type = {}
        _inner(optuna_params, new)
        return new

    @staticmethod
    def _convert_ema_decay(value: Any) -> optuna_params_type:
        prefix = value
        use_ema_key = f"{prefix}_use_ema"
        ema_decay_key = f"{prefix}_ema_decay"
        return {
            "use_ema": OptunaParam(use_ema_key, [True, False], "categorical"),
            "ema_decay": OptunaParam(ema_decay_key, [0.0, 1.0], "float"),
        }

    @staticmethod
    def _convert_clip_norm(value: Any) -> optuna_params_type:
        prefix = value
        use_clip_norm_key = f"{prefix}_use_clip_norm"
        clip_norm_key = f"{prefix}_clip_norm"
        ucn_param = OptunaParam(use_clip_norm_key, [True, False], "categorical")
        return {
            "use_clip_norm": ucn_param,
            "clip_norm": OptunaParam(clip_norm_key, [0.25, 1.25], "float"),
        }

    @staticmethod
    def _convert_dropout(value: Any) -> optuna_params_type:
        prefix = value
        use_dropout_key = f"{prefix}_use_dropout"
        drop_ratio_key = f"{prefix}_drop_ratio"
        return {
            "use_dropout": OptunaParam(use_dropout_key, [True, False], "categorical"),
            "drop_ratio": OptunaParam(drop_ratio_key, [0.1, 0.9], "float"),
        }

    @staticmethod
    def _convert_hidden_units(value: Any) -> optuna_params_type:
        # parse
        config: Dict[str, Any] = {}
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
        rs: optuna_params_type = {
            "num_layers": OptunaParam(num_layers_name, [1, num_layers], "int")
        }
        for i in range(num_layers):
            key = f"hidden_unit_{i}"
            rs[key] = OptunaParam(f"{prefix}_{key}", [low, high], "int", config)
        return rs

    @staticmethod
    def _convert_dndf_config(value: Any) -> optuna_params_type:
        split = value.split("_")
        if len(split) == 3:
            force = False
            prefix, num_tree, tree_depth = split
        else:
            force = True
            assert split[-1] == "force"
            prefix, num_tree, tree_depth = split[:-1]
        num_tree, tree_depth = map(int, [num_tree, tree_depth])
        num_tree = max(4, num_tree)
        tree_depth = max(2, tree_depth)
        num_tree_key = f"{prefix}_num_tree"
        tree_depth_key = f"{prefix}_tree_depth"
        config: optuna_params_type = {
            "num_tree": OptunaParam(num_tree_key, [4, num_tree], "int", {"log": True}),
            "tree_depth": OptunaParam(tree_depth_key, [2, tree_depth], "int"),
        }
        if not force:
            use_dndf_key = f"{prefix}_use_dndf"
            use_dndf_param = OptunaParam(use_dndf_key, [True, False], "categorical")
            config[use_dndf_key] = use_dndf_param
        return config

    @staticmethod
    def _convert_pruner_config(value: Any) -> optuna_params_type:
        prefix = value
        available_methods = ["auto_prune", "surgery", "simplified"]
        use_pruner_key = f"{prefix}_use_pruner"
        method_key = f"{prefix}_prune_method"
        return {
            "use_pruner": OptunaParam(use_pruner_key, [True, False], "categorical"),
            "prune_method": OptunaParam(method_key, available_methods, "categorical"),
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
    def _parse_ema_decay(d: Dict[str, Any], trial: Trial) -> Any:
        use_ema = d["use_ema"]
        if trial is not None:
            use_ema = use_ema.pop(trial)
        if not use_ema:
            return 0.0
        ema_decay = d["ema_decay"]
        if trial is not None:
            ema_decay = ema_decay.pop(trial)
        return ema_decay

    @staticmethod
    def _parse_clip_norm(d: Dict[str, Any], trial: Trial) -> Any:
        use_clip_norm = d["use_clip_norm"]
        if trial is not None:
            use_clip_norm = use_clip_norm.pop(trial)
        if not use_clip_norm:
            return 0.0
        clip_norm = d["clip_norm"]
        if trial is not None:
            clip_norm = clip_norm.pop(trial)
        return clip_norm

    @staticmethod
    def _parse_dropout(d: Dict[str, Any], trial: Trial) -> Any:
        use_dropout = d["use_dropout"]
        if trial is not None:
            use_dropout = use_dropout.pop(trial)
        if not use_dropout:
            return 0.0
        drop_ratio = d["drop_ratio"]
        if trial is not None:
            drop_ratio = drop_ratio.pop(trial)
        return drop_ratio

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
        use_dndf = d.get("use_dndf", None)
        if use_dndf is None:
            use_dndf = True
        else:
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
        method = d["prune_method"]
        if trial is not None:
            method = method.pop(trial)
        return {"method": method}

    @staticmethod
    def _make_prefix(prefix: str) -> str:
        return f"[{prefix}]" if prefix else ""

    # api

    @classmethod
    def merge_user_prefix(cls, k: str, user_prefix: Optional[str]) -> str:
        if user_prefix is None:
            return k
        return f"{user_prefix}_{k}"

    @classmethod
    def make_ema_decay(cls, prefix: str) -> Dict[str, str]:
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}ema_decay]"
        value = prefix
        return {key: value}

    @classmethod
    def make_clip_norm(cls, prefix: str) -> Dict[str, str]:
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}clip_norm]"
        value = prefix
        return {key: value}

    @classmethod
    def make_dropout(cls, prefix: str) -> Dict[str, str]:
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}dropout]"
        value = prefix
        return {key: value}

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
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}hidden_units]"
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
        force: bool,
    ) -> Dict[str, str]:
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}dndf_config]"
        value = f"{prefix}_{num_tree}_{tree_depth}"
        if force:
            value = f"{value}_force"
        return {key: value}

    @classmethod
    def make_pruner_config(cls, prefix: str) -> Dict[str, str]:
        key = f"{cls.prefix}[{cls._make_prefix(prefix)}pruner_config]"
        value = prefix
        return {key: value}


class OptunaKeyMapping(LoggingMixin):
    tuner_folder = "__tuner__"
    optuna_params_name = "__optuna_params__"

    def __init__(self, tuner: _Tuner, optuna_params: optuna_params_type):
        self.tuner = tuner
        self.delim = SAVING_DELIM
        self.converter = OptunaParamConverter()
        self.params = self.converter.convert(optuna_params)
        self.optuna_key_mapping: Dict[str, str] = {}

        def _inject_mapping(d: optuna_params_type, prefix_list: List[str]) -> None:
            for k, v in d.items():
                new_prefix_list = prefix_list + [k]
                if isinstance(v, OptunaParam):
                    self.optuna_key_mapping[v.name] = self.delim.join(new_prefix_list)
                    continue
                assert isinstance(v, dict)
                _inject_mapping(v, new_prefix_list)

        _inject_mapping(self.params, [])
        self._optuna_params = optuna_params

    def pop(self, trial: Trial) -> Dict[str, Any]:
        optuna_params = shallow_copy_dict(self.params)
        current_params = shallow_copy_dict(self.tuner.base_params)

        def _inject_suggestion(d: optuna_params_type, current: Dict[str, Any]) -> None:
            for k, v in d.items():
                usage, user_prefix = self.converter.get_usage(k)
                if usage is not None:
                    assert isinstance(v, dict)
                    k = OptunaParamConverter.merge_user_prefix(usage, user_prefix)
                    current[k] = self.converter.pop(usage, v, trial)
                    continue
                if isinstance(v, dict):
                    _inject_suggestion(v, current.setdefault(k, {}))
                    continue
                assert isinstance(v, OptunaParam)
                current[k] = v.pop(trial)

        _inject_suggestion(optuna_params, current_params)
        return current_params

    def parse(self, optuna_param_values: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for k, v in optuna_param_values.items():
            key_mapping = self.optuna_key_mapping[k]
            key_path = key_mapping.split(self.delim)
            local_param = params
            for sub_k in key_path[:-1]:
                local_param = local_param.setdefault(sub_k, {})
            local_param[key_path[-1]] = v
        return params

    def convert(self, param_values: Dict[str, Any]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}

        def _inject_values(d: Dict[str, Any], current: dict) -> None:
            for k, v in d.items():
                usage, user_prefix = self.converter.get_usage(k)
                if usage is not None:
                    current[usage] = self.converter.parse(usage, v)
                    continue
                if isinstance(v, dict):
                    _inject_values(v, current.setdefault(k, {}))
                    continue
                current[k] = v

        _inject_values(param_values, converted)
        return converted

    def save(self, export_folder: str) -> "OptunaKeyMapping":
        Saving.prepare_folder(self, export_folder)
        tuner_folder = os.path.join(export_folder, self.tuner_folder)
        self.tuner.save(tuner_folder)
        Saving.save_dict(self._optuna_params, self.optuna_params_name, export_folder)
        return self

    @classmethod
    def load(cls, export_folder: str) -> "OptunaKeyMapping":
        tuner_folder = os.path.join(export_folder, cls.tuner_folder)
        tuner = _Tuner.load(tuner_folder)
        optuna_params = Saving.load_dict(cls.optuna_params_name, export_folder)
        return cls(tuner, optuna_params)


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
    def __init__(
        self,
        *,
        tune_lr: bool = True,
        tune_optimizer: bool = True,
        tune_scheduler: bool = True,
        tune_ema_decay: bool = True,
        tune_clip_norm: bool = True,
        tune_batch_size: bool = True,
        tune_init_method: bool = True,
        **kwargs: Any,
    ) -> None:
        self.base_params: optuna_params_type = {}
        if tune_lr:
            lr_param = OptunaParam("lr", [1e-5, 0.1], "float", {"log": True})
            optimizer_config = self.base_params.setdefault("optimizer_config", {})
            assert isinstance(optimizer_config, dict)
            optimizer_config["lr"] = lr_param
        if tune_optimizer:
            optimizer_param = OptunaParam(
                "optimizer",
                ["nag", "rmsprop", "adam", "adamw"],
                "categorical",
            )
            self.base_params["optimizer"] = optimizer_param
        if tune_scheduler:
            scheduler_param = OptunaParam(
                "scheduler",
                ["step", "plateau", "warmup"],
                "categorical",
            )
            self.base_params["scheduler"] = scheduler_param
        if tune_ema_decay:
            model_config = self.base_params.setdefault("model_config", {})
            assert isinstance(model_config, dict)
            model_config.update(OptunaParamConverter.make_ema_decay(""))
        if tune_clip_norm:
            trainer_config = self.base_params.setdefault("trainer_config", {})
            assert isinstance(trainer_config, dict)
            trainer_config.update(OptunaParamConverter.make_clip_norm(""))
        if tune_batch_size:
            choices = [16, 32, 64, 128, 256, 512, 1024]
            bs_param = OptunaParam("batch_size", choices, "categorical")
            self.base_params["batch_size"] = bs_param
        if tune_init_method:
            default_encoding_init_param = OptunaParam(
                "default_encoding_init_method",
                [None, "truncated_normal"],
                "categorical",
            )
            model_config = self.base_params.setdefault("model_config", {})
            assert isinstance(model_config, dict)
            de_cfg = model_config.setdefault("default_encoding_configs", {})
            de_cfg["init_method"] = default_encoding_init_param
        self.kwargs = shallow_copy_dict(kwargs)

    def get(self, model: str) -> optuna_params_type:
        attr = getattr(self, f"_{model}_preset", None)
        if attr is None:
            raise NotImplementedError(f"preset params for '{model}' is not defined")
        preset = attr()
        if not preset:
            raise ValueError("current preset params is empty")
        return preset

    def _nnb_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)

    def _ndt_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)

    def _linear_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)

    @staticmethod
    def _get_head_config(params: Dict[str, Any], pipe: str) -> Dict[str, Any]:
        model_config = params.setdefault("model_config", {})
        pipe_configs = model_config.setdefault("pipe_configs", {})
        return pipe_configs.setdefault(pipe, {}).setdefault("head", {})

    def _fcnn_preset(self) -> optuna_params_type:
        params = shallow_copy_dict(self.base_params)
        if self.kwargs.get("tune_hidden_units", True):
            hu_param = OptunaParamConverter.make_hidden_units("", 8, 2048, 3)
            head_config = self._get_head_config(params, "fcnn")
            head_config.update(hu_param)
        mapping_config: optuna_params_type = {}
        if self.kwargs.get("tune_batch_norm", True):
            bn_param = OptunaParam("mlp_batch_norm", [False, True], "categorical")
            mapping_config["batch_norm"] = bn_param
        if self.kwargs.get("tune_dropout", True):
            mapping_config.update(OptunaParamConverter.make_dropout(""))
        if self.kwargs.get("tune_pruner", True):
            mapping_config.update(OptunaParamConverter.make_pruner_config(""))
        if mapping_config:
            head_config = self._get_head_config(params, "fcnn")
            head_config["mapping_configs"] = mapping_config
        if self.kwargs.get("tune_embedding_dim", True):
            ed_param = OptunaParam("embedding_dim", [8, "auto"], "categorical")
            model_config = params.setdefault("model_config", {})
            assert isinstance(model_config, dict)
            model_config["default_encoding_configs"]["embedding_dim"] = ed_param
        return params

    def _tree_dnn_preset(self) -> optuna_params_type:
        params = self._fcnn_preset()
        if self.kwargs.get("tune_dndf", True):
            dndf_param = OptunaParamConverter.make_dndf_config("", 32, 4, False)
            head_config = self._get_head_config(params, "dndf")
            head_config.update(dndf_param)
        return params

    def _tree_linear_preset(self) -> optuna_params_type:
        params = shallow_copy_dict(self.base_params)
        if self.kwargs.get("tune_dndf", True):
            dndf_param = OptunaParamConverter.make_dndf_config("out", 64, 4, True)
            head_config = self._get_head_config(params, "tree_stack")
            head_config.update(dndf_param)
        return params

    def _tree_stack_preset(self) -> optuna_params_type:
        params = shallow_copy_dict(self.base_params)
        if self.kwargs.get("tune_num_blocks", True):
            head_config = self._get_head_config(params, "tree_stack")
            head_config["num_blocks"] = OptunaParam("num_blocks", [1, 3], "int")
        if self.kwargs.get("tune_inner_dndf", True):
            head_config = self._get_head_config(params, "tree_stack")
            inner_param = OptunaParamConverter.make_dndf_config("", 64, 4, True)
            head_config.update(inner_param)
        if self.kwargs.get("tune_dndf", True):
            head_config = self._get_head_config(params, "tree_stack")
            out_param = OptunaParamConverter.make_dndf_config("out", 32, 4, True)
            head_config.update(out_param)
        return params

    # TODO : optimize these three preset

    def _ddr_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)

    def _ddr_q_preset(self) -> optuna_params_type:
        return self._ddr_preset()

    def _ddr_cdf_preset(self) -> optuna_params_type:
        return self._ddr_preset()

    def _rnn_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)

    def _transformer_preset(self) -> optuna_params_type:
        return shallow_copy_dict(self.base_params)


class OptunaArgs(NamedTuple):
    cuda: Optional[str]
    compress: bool
    num_trial: Union[str, int]
    task_config: Union[str, Dict[str, Any]]
    key_mapping: Union[str, OptunaKeyMapping]


def optuna_core(args: OptunaArgs) -> optuna.study.Study:
    cuda = args.cuda
    if cuda == "None":
        cuda = None
    compress = args.compress

    key_mapping_arg = args.key_mapping
    if isinstance(key_mapping_arg, str):
        key_mapping = OptunaKeyMapping.load(key_mapping_arg)
    else:
        assert isinstance(key_mapping_arg, OptunaKeyMapping)
        key_mapping = key_mapping_arg
    tuner = key_mapping.tuner

    config = args.task_config
    if isinstance(config, dict):
        config_dict = config
    else:
        config_dict = Saving.load_dict("config", config)
    model = config_dict["model"]
    metrics = config_dict["metrics"]
    timeout = config_dict["timeout"]
    num_jobs = config_dict["num_jobs"]
    num_repeat = config_dict["num_repeat"]
    num_parallel = config_dict["num_parallel"]
    estimator_scoring_function = config_dict["estimator_scoring_function"]
    study_config = config_dict["study_config"]
    temp_folder = config_dict["temp_folder"]

    def objective(trial: Trial) -> float:
        temp_folder_ = os.path.join(temp_folder, str(trial.number))
        current_params = key_mapping.pop(trial)
        if metrics is not None:
            current_params["metrics"] = metrics
        inject_mlflow_stuffs(
            model,
            config=current_params,
            run_name_prefix=f"trial_{trial.number}",
        )
        args_ = model, current_params, num_repeat, num_parallel, temp_folder_
        sequential = None if num_jobs <= 1 else True
        result = tuner.train(
            *args_,
            sequential=sequential,
            compress=compress,
            cuda=cuda,
        )
        final_scores = []
        weighted_metrics = result.weighted_metrics
        estimators = tuner.make_estimators(metrics, result.repeat_result)
        for estimator in estimators:
            scoring_fn = estimator.scoring_fn(estimator_scoring_function)
            final_scores.append(scoring_fn(weighted_metrics[estimator.type]))
        return sum(final_scores) / len(final_scores)

    study = optuna.create_study(**study_config)
    study.optimize(objective, int(args.num_trial), timeout, 1)
    return study


def optuna_tune(
    x: data_type = None,
    y: data_type = None,
    x_cv: data_type = None,
    y_cv: data_type = None,
    *,
    model: str = "fcnn",
    task_type: task_type_type = "",
    tuner: Optional[_Tuner] = None,
    params: Optional[optuna_params_type] = None,
    study_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    num_jobs: int = 1,
    num_trial: int = 50,
    num_repeat: int = 5,
    num_parallel: int = 1,
    compress: bool = True,
    temp_folder: str = "__tmp__",
    estimator_scoring_function: Union[str, scoring_fn_type] = default_scoring,
    timeout: Optional[float] = None,
    score_weights: Optional[Dict[str, float]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    cuda: Optional[Union[int, str]] = None,
) -> OptunaResult:
    if params is None:
        params = OptunaPresetParams().get(model)

    if tuner is None:
        if x is None:
            raise ValueError("either `x` or `tuner` should be provided")
        extra_config = _init_extra_config(metrics, score_weights, extra_config)
        tuner = _Tuner(x, y, x_cv, y_cv, task_type, **extra_config)
    key_mapping = OptunaKeyMapping(tuner, params)

    if num_jobs <= 1:
        meta_folder = None
    else:
        if cuda is not None:
            print(
                f"{LoggingMixin.warning_prefix}`cuda` is set to {cuda} "
                "but will take no effect because `num_jobs` is set "
                f"to be greater than 1 ({num_jobs})"
            )
        meta_folder = os.path.join(temp_folder, "__meta__")

    if study_config is None:
        study_config = {}
    if num_jobs <= 1:
        storage = None
    else:
        assert isinstance(meta_folder, str)
        os.makedirs(meta_folder, exist_ok=True)
        storage = os.path.join(meta_folder, "shared.db")
        storage = f"sqlite:///{storage}"
    study_config["storage"] = storage
    study_config["direction"] = "maximize"
    study_config["load_if_exists"] = True
    study_config.setdefault("study_name", f"{model}_optuna")
    if num_jobs > 1:
        optuna.create_study(**study_config)

    task_config = {
        "model": model,
        "metrics": metrics,
        "timeout": timeout,
        "num_jobs": num_jobs,
        "num_repeat": num_repeat,
        "num_parallel": num_parallel,
        "estimator_scoring_function": estimator_scoring_function,
        "study_config": study_config,
        "temp_folder": temp_folder,
    }

    if num_jobs <= 1:
        if isinstance(cuda, int):
            cuda = str(cuda)
        args = OptunaArgs(cuda, compress, num_trial, task_config, key_mapping)
        study = optuna_core(args)
    else:
        assert isinstance(meta_folder, str)

        key_mapping_folder = os.path.join(meta_folder, "__key_mapping__")
        key_mapping.save(key_mapping_folder)

        task_config_folder = os.path.join(meta_folder, "__task_config__")
        os.makedirs(task_config_folder, exist_ok=True)
        Saving.save_dict(task_config, "config", task_config_folder)

        experiment = Experiment(num_jobs=num_jobs)
        num_trials = [num_trial // num_jobs] * num_jobs
        num_trials[-1] = num_trial - sum(num_trials[:-1])
        for n in num_trials:
            experiment.add_task(
                execute="optuna",
                root_workplace=temp_folder,
                num_trial=n,
                compress=compress,
                task_config_folder=task_config_folder,
                key_mapping_folder=key_mapping_folder,
            )
        experiment.run_tasks()
        study = optuna.create_study(**study_config)

    return OptunaResult(tuner, study, key_mapping)


__all__ = [
    "tune_with",
    "optuna_core",
    "optuna_tune",
    "OptunaParam",
    "OptunaParamConverter",
    "OptunaKeyMapping",
    "OptunaPresetParams",
]
