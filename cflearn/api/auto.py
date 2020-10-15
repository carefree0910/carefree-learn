import os
import optuna

import numpy as np
import optuna.visualization as vis

from typing import *
from cftool.ml.utils import *
from cfdata.tabular import *
from optuna.importance import BaseImportanceEvaluator
from plotly.graph_objects import Figure

from .basic import *
from .ensemble import *
from ..misc.toolkit import *
from .hpo import optuna_tune


class Auto:
    def __init__(self, task_type: TaskTypes, *, model: str = "fcnn"):
        self.model = model
        self.task_type = task_type

    def __str__(self):
        return f"Auto_{self.model}({self.task_type})"

    __repr__ = __str__

    @property
    def study(self) -> optuna.study.Study:
        return self.optuna_result.study

    @property
    def predict(self) -> Callable:
        return self.pattern.predict

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

    # api

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
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
