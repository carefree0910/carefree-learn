import os
import optuna

import numpy as np
import optuna.visualization as vis

from typing import *
from cfdata.tabular import TaskTypes
from cftool.misc import shallow_copy_dict
from cftool.ml.utils import scoring_fn_type
from optuna.trial import TrialState, FrozenTrial
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
        study_config: Dict[str, Any] = None,
        metrics: Union[str, List[str]] = None,
        num_jobs: int = 1,
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
        cuda: Union[str, int] = None,
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
            bagging_config["use_tracker"] = False
            bagging_config["num_jobs"] = num_jobs
            bagging_config["models"] = model
            bagging_config["k"] = num_final_repeat
            x_merged, y_merged = self._merge_data(x, y, x_cv, y_cv)
            self.bagging_result = bagging(x_merged, y_merged, **bagging_config)
            self.pattern = self.bagging_result.pattern
            self.data = self.bagging_result.data
        else:
            repeat_temp_folder = os.path.join(temp_folder, "__repeat__")
            repeat_config = shallow_copy_dict(self.best_param)
            repeat_config.update(
                {
                    "models": model,
                    "sequential": num_jobs <= 1,
                    "temp_folder": repeat_temp_folder,
                    "num_repeat": num_final_repeat,
                    "num_jobs": num_jobs,
                }
            )
            self.repeat_result = repeat_with(x, y, x_cv, y_cv, **repeat_config)
            patterns_dict = self.repeat_result.patterns
            assert patterns_dict is not None
            patterns = patterns_dict[model]
            data = self.repeat_result.data
            self.pattern = ensemble(patterns)
            self.data = data
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
