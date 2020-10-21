import os
import torch
import optuna

import numpy as np
import optuna.visualization as vis

from typing import *
from functools import partial
from cfdata.tabular import TaskTypes
from cfdata.tabular import TabularData
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.ml.utils import scoring_fn_type
from optuna.trial import TrialState
from optuna.trial import FrozenTrial
from optuna.importance import BaseImportanceEvaluator
from plotly.graph_objects import Figure

from .basic import *
from .ensemble import *
from .hpo import optuna_tune
from .hpo import optuna_params_type
from .production import Pack
from .production import Predictor
from ..types import data_type
from ..pipeline.core import Pipeline


class Auto:
    data_folder = "__data__"
    pattern_weights_file = "pattern_weights.npy"

    def __init__(self, task_type: TaskTypes, *, model: str = "fcnn"):
        self.model = model
        self.task_type = task_type
        self.pipelines: Optional[List[Pipeline]] = None
        self.pattern_weights: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"Auto_{self.model}({self.task_type})"

    __repr__ = __str__

    @property
    def study(self) -> optuna.study.Study:
        return self.optuna_result.study

    @property
    def predict(self) -> Callable:
        return self.pattern.predict

    @property
    def predict_prob(self) -> Callable:
        return partial(self.pattern.predict, requires_prob=True)

    @property
    def pruned_trials(self) -> List[FrozenTrial]:
        return [t for t in self.study.trials if t.state == TrialState.PRUNED]

    @property
    def complete_trials(self) -> List[FrozenTrial]:
        return [t for t in self.study.trials if t.state == TrialState.COMPLETE]

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
                assert isinstance(x, list)
                x_merged = x + x_cv
                if y is None:
                    y_merged = None
                else:
                    assert isinstance(y, list) and isinstance(y_cv, list)
                    y_merged = y + y_cv
            else:
                x_merged = np.vstack([x, x_cv])
                y_merged = None if y is None else np.vstack([y, y_cv])
            return x_merged, y_merged
        assert isinstance(x, str)
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
        return new_file, None

    # api

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        params: optuna_params_type = None,
        study_config: Dict[str, Any] = None,
        predict_config: Optional[Dict[str, Any]] = None,
        metrics: Union[str, List[str]] = None,
        num_jobs: int = 1,
        num_trial: int = 50,
        num_repeat: int = 5,
        num_parallel: int = 0,
        timeout: Optional[float] = None,
        score_weights: Optional[Dict[str, float]] = None,
        estimator_scoring_function: Union[str, scoring_fn_type] = "default",
        temp_folder: str = "__tmp__",
        num_final_repeat: int = 10,
        bagging_config: Optional[Dict[str, Any]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
        cuda: Optional[Union[str, int]] = None,
    ) -> "Auto":
        model = self.model
        optuna_temp_folder = os.path.join(temp_folder, "__optuna__")
        self.optuna_result = optuna_tune(
            x,
            y,
            x_cv,
            y_cv,
            model=model,
            task_type=self.task_type,
            params=params,
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
            cuda=cuda,
        )
        num_jobs = max(num_jobs, num_parallel)
        self.best_param = self.optuna_result.best_param
        self.repeat_result: Optional[RepeatResult] = None
        self.bagging_result: Optional[EnsembleResults] = None
        if bagging_config is not None:
            bagging_temp_folder = os.path.join(temp_folder, "__bagging__")
            bagging = Ensemble(self.task_type, self.best_param).bagging
            bagging_config.setdefault("task_name", f"{model}_opt")
            increment_config = bagging_config.setdefault("increment_config", {})
            increment_config.setdefault("trigger_logging", False)
            increment_config.setdefault("verbose_level", 0)
            bagging_config["temp_folder"] = bagging_temp_folder
            bagging_config["predict_config"] = predict_config
            bagging_config["use_tracker"] = False
            bagging_config["num_jobs"] = num_jobs
            bagging_config["models"] = model
            bagging_config["k"] = num_final_repeat
            x_merged, y_merged = self._merge_data(x, y, x_cv, y_cv)
            self.bagging_result = bagging(x_merged, y_merged, **bagging_config)
            self.pattern_weights = self.bagging_result.pattern_weights
            self.pipelines = self.bagging_result.pipelines
            self.pattern = self.bagging_result.pattern
            self.data = self.bagging_result.data
        else:
            repeat_temp_folder = os.path.join(temp_folder, "__repeat__")
            repeat_config = shallow_copy_dict(self.best_param)
            repeat_config.update(
                {
                    "models": model,
                    "sequential": num_jobs <= 1,
                    "predict_config": predict_config,
                    "temp_folder": repeat_temp_folder,
                    "num_repeat": num_final_repeat,
                    "num_jobs": num_jobs,
                }
            )
            x, y, x_cv, y_cv = self.optuna_result.tuner.make_data()
            self.repeat_result = repeat_with(x, y, x_cv, y_cv, **repeat_config)
            patterns_dict = self.repeat_result.patterns
            pipelines_dict = self.repeat_result.pipelines
            assert patterns_dict is not None and pipelines_dict is not None
            data = self.repeat_result.data
            self.pipelines = pipelines_dict[model]
            self.pattern = ensemble(patterns_dict[model])
            self.data = data
        return self

    def pack(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
    ) -> "Auto":
        if self.pipelines is None:
            raise ValueError("`pipelines` are not yet generated")
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            data_folder = os.path.join(export_folder, self.data_folder)
            first_inference = self.pipelines[0].inference
            if first_inference is None:
                raise ValueError("`inference` in pipeline is not yet generated")
            first_inference.data.save(
                data_folder,
                retain_data=retain_data,
                compress=False,
            )
            for i, pipeline in enumerate(self.pipelines):
                local_export_folder = os.path.join(export_folder, f"m_{i:04d}")
                Pack.pack(
                    pipeline,
                    local_export_folder,
                    pack_data=False,
                    compress=False,
                )
            if self.pattern_weights is not None:
                path = os.path.join(export_folder, self.pattern_weights_file)
                np.save(path, self.pattern_weights)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def get_predictors(
        cls,
        export_folder: str,
        device: Union[str, torch.device] = "cpu",
        *,
        compress: bool = True,
        use_tqdm: bool = False,
    ) -> Tuple[List[Predictor], Optional[np.ndarray]]:
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(
                export_folder,
                compress,
                remove_extracted=True,
            ):
                data_folder = os.path.join(export_folder, cls.data_folder)
                data = TabularData.load(data_folder, compress=False)
                predictors = []
                pattern_weights = None
                for stuff in os.listdir(export_folder):
                    if stuff == cls.pattern_weights_file:
                        pattern_weights = np.load(os.path.join(export_folder, stuff))
                    else:
                        if stuff == cls.data_folder:
                            continue
                        local_folder = os.path.join(export_folder, stuff)
                        predictors.append(
                            Pack.get_predictor(
                                local_folder,
                                device,
                                data=data,
                                compress=False,
                                use_tqdm=use_tqdm,
                            )
                        )
        return predictors, pattern_weights

    # visualization

    def plot_param_importances(
        self,
        evaluator: BaseImportanceEvaluator = None,
        params: List[str] = None,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_param_importances(self.study, evaluator, params)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "param_importances.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_contour(
        self,
        params: List[str] = None,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_contour(self.study, params)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "contour.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_parallel_coordinate(
        self,
        params: List[str] = None,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_parallel_coordinate(self.study, params)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "parallel_coordinate.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_slice(
        self,
        params: List[str] = None,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_slice(self.study, params)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "slice.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_optimization_history(
        self,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_optimization_history(self.study)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "optimization_history.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_intermediate_values(
        self,
        export_folder: str = None,
    ) -> Figure:
        fig = vis.plot_intermediate_values(self.study)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "intermediate_values.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig

    def plot_edf(self, export_folder: str = None) -> Figure:
        fig = vis.plot_edf(self.study)
        if export_folder is not None:
            os.makedirs(export_folder, exist_ok=True)
            html_path = os.path.join(export_folder, "edf.html")
            with open(html_path, "w") as f:
                f.write(fig.to_html())
        return fig


__all__ = ["Auto"]
