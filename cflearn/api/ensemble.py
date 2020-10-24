import os
import math
import tqdm
import torch

import numpy as np

from typing import *
from cftool.misc import hash_code
from cftool.misc import update_dict
from cftool.misc import lock_manager
from cftool.misc import shallow_copy_dict
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cftool.ml.utils import collate_fn_type
from cftool.ml.utils import Metrics
from cftool.ml.utils import Comparer
from cfdata.tabular import task_type_type
from cfdata.tabular import parse_task_type
from cfdata.tabular import KFold
from cfdata.tabular import KRandom
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset
from torch.nn.functional import one_hot

from .zoo import zoo
from .basic import *
from ..dist import *
from ..misc.toolkit import *
from .register import register_metric
from ..types import data_type
from ..pipeline.core import Pipeline


class BenchmarkResults(NamedTuple):
    data: TabularData
    best_configs: Dict[str, Dict[str, Any]]
    best_methods: Dict[str, str]
    experiments: Experiments
    comparer: Comparer


class Benchmark(LoggingMixin):
    def __init__(
        self,
        task_name: str,
        task_type: task_type_type,
        *,
        temp_folder: Optional[str] = None,
        project_name: str = "carefree-learn",
        models: Union[str, List[str]] = "fcnn",
        increment_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        use_tracker: bool = False,
        use_cuda: bool = True,
    ):
        self.data: Optional[TabularData] = None
        self.experiments: Optional[Experiments] = None

        if data_config is None:
            data_config = {}
        if read_config is None:
            read_config = {}
        self.data_config, self.read_config = data_config, read_config
        self.task_name = task_name
        self.task_type = task_type
        if temp_folder is None:
            temp_folder = f"__{task_name}__"
        self.temp_folder, self.project_name = temp_folder, project_name
        if isinstance(models, str):
            models = [models]
        self.models = models
        if increment_config is None:
            increment_config = {}
        self.increment_config = increment_config
        self.use_tracker = use_tracker
        self.use_cuda = use_cuda

    @property
    def identifier(self) -> str:
        return hash_code(
            f"{self.project_name}{self.task_name}{self.models}{self.increment_config}"
        )

    @property
    def data_tasks(self) -> List[Optional[Task]]:
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` are not yet generated")
        return next(iter(experiments.data_tasks.values()))

    def _add_tasks(
        self,
        iterator_name: str,
        data_tasks: List[Task],
        experiments: Experiments,
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> None:
        self.configs: Dict[str, Dict[str, Any]] = {}
        for i in range(len(data_tasks)):
            for model in self.models:
                model_benchmarks = benchmarks.get(model)
                if model_benchmarks is None:
                    model_benchmarks = zoo(model).benchmarks
                for model_type, config in model_benchmarks.items():
                    identifier = f"{model}_{self.task_name}_{model_type}"
                    task_name = f"{identifier}_{iterator_name}{i}"
                    increment_config = shallow_copy_dict(self.increment_config)
                    config = update_dict(increment_config, config)
                    self.configs.setdefault(identifier, config)
                    if not self.use_tracker:
                        tracker_config = None
                    else:
                        tracker_config = {
                            "project_name": self.project_name,
                            "task_name": task_name,
                            "overwrite": True,
                        }
                    experiments.add_task(
                        model=model,
                        data_task=data_tasks[i],
                        identifier=identifier,
                        tracker_config=tracker_config,
                        **config,
                    )

    def _run_tasks(
        self,
        num_jobs: int = 4,
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` is not yet defined")
        results: Dict[str, List[Any]] = experiments.run_tasks(
            num_jobs=num_jobs,
            run_tasks=run_tasks,
            load_task=load_task,
        )
        comparer_list = []
        for i, data_task in enumerate(self.data_tasks):
            assert data_task is not None
            pipelines = {}
            x_te, y_te = data_task.fetch_data("_te")
            for identifier, ms in results.items():
                pipelines[identifier] = ms[i]
            comparer = estimate(
                x_te,
                y_te,
                pipelines=pipelines,
                predict_config=predict_config,
                comparer_verbose_level=None,
            )
            comparer_list.append(comparer)
        comparer = Comparer.merge(comparer_list)
        best_methods = comparer.best_methods
        best_configs = {
            metric: self.configs[identifier]
            for metric, identifier in best_methods.items()
        }
        return BenchmarkResults(
            self.data,
            best_configs,
            best_methods,
            experiments,
            comparer,
        )

    def _pre_process(self, x: data_type, y: data_type = None) -> TabularDataset:
        data_config = shallow_copy_dict(self.data_config)
        task_type = data_config.pop("task_type", None)
        if task_type is not None:
            assert parse_task_type(task_type) is parse_task_type(self.task_type)
        self.data = TabularData.simple(self.task_type, **data_config)
        self.data.read(x, y, **self.read_config)
        return self.data.to_dataset()

    def _k_core(
        self,
        k_iterator: Iterable,
        num_jobs: int,
        run_tasks: bool,
        predict_config: Optional[Dict[str, Any]],
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    ) -> BenchmarkResults:
        if benchmarks is None:
            benchmarks = {}
        self.experiments = Experiments(self.temp_folder, use_cuda=self.use_cuda)
        data_tasks = []
        for i, (train_split, test_split) in enumerate(k_iterator):
            train_dataset, test_dataset = train_split.dataset, test_split.dataset
            x_tr, y_tr = train_dataset.xy
            x_te, y_te = test_dataset.xy
            data_task = Task.data_task(i, self.identifier, self.experiments)
            data_task.dump_data(x_tr, y_tr)
            data_task.dump_data(x_te, y_te, "_te")
            data_tasks.append(data_task)
        self._iterator_name = type(k_iterator).__name__
        self._add_tasks(self._iterator_name, data_tasks, self.experiments, benchmarks)
        return self._run_tasks(num_jobs, run_tasks, predict_config)

    def k_fold(
        self,
        k: int,
        x: data_type,
        y: data_type = None,
        *,
        num_jobs: int = 4,
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(
            KFold(k, dataset),
            num_jobs,
            run_tasks,
            predict_config,
            benchmarks,
        )

    def k_random(
        self,
        k: int,
        num_test: Union[int, float],
        x: data_type,
        y: data_type = None,
        *,
        num_jobs: int = 4,
        run_tasks: bool = True,
        predict_config: Optional[Dict[str, Any]] = None,
        benchmarks: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(
            KRandom(k, num_test, dataset),
            num_jobs,
            run_tasks,
            predict_config,
            benchmarks,
        )

    def save(
        self,
        export_folder: str,
        *,
        simplify: bool = True,
        compress: bool = True,
    ) -> "Benchmark":
        experiments = self.experiments
        if experiments is None:
            raise ValueError("`experiments` is not yet defined")
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            if isinstance(self.task_type, str):
                task_type_value = self.task_type
            else:
                task_type_value = self.task_type.value
            Saving.save_dict(
                {
                    "task_name": self.task_name,
                    "task_type": task_type_value,
                    "project_name": self.project_name,
                    "models": self.models,
                    "increment_config": self.increment_config,
                    "use_cuda": self.use_cuda,
                    "iterator_name": self._iterator_name,
                    "temp_folder": self.temp_folder,
                    "configs": self.configs,
                },
                "kwargs",
                abs_folder,
            )
            experiments_folder = os.path.join(abs_folder, "__experiments__")
            experiments.save(
                experiments_folder,
                simplify=simplify,
                compress=compress,
            )
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        saving_folder: str,
        *,
        predict_config: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> Tuple["Benchmark", BenchmarkResults]:
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            with Saving.compress_loader(abs_folder, compress, remove_extracted=False):
                kwargs = Saving.load_dict("kwargs", abs_folder)
                configs = kwargs.pop("configs")
                iterator_name = kwargs.pop("iterator_name")
                benchmark = cls(**shallow_copy_dict(kwargs))
                benchmark.configs = configs
                benchmark._iterator_name = iterator_name
                benchmark.experiments = Experiments.load(
                    os.path.join(abs_folder, "__experiments__")
                )
                results = benchmark._run_tasks(0, False, predict_config)
        return benchmark, results


def ensemble(
    patterns: List[ModelPattern],
    *,
    pattern_weights: Optional[np.ndarray] = None,
    ensemble_method: Optional[Union[str, collate_fn_type]] = None,
) -> EnsemblePattern:
    if ensemble_method is None:
        if pattern_weights is None:
            ensemble_method = "default"
        else:
            if abs(pattern_weights.sum() - 1.0) > 1e-4:
                raise ValueError("`pattern_weights` should sum to 1.0")
            pattern_weights = pattern_weights.reshape([-1, 1, 1])

            def ensemble_method(
                arrays: List[np.ndarray],
                requires_prob: bool,
            ) -> np.ndarray:
                predictions = np.array(arrays).reshape(
                    [len(arrays), len(arrays[0]), -1]
                )
                if requires_prob or not is_int(predictions):
                    return (predictions * pattern_weights).sum(axis=0)
                encodings = one_hot(to_torch(predictions).to(torch.long).squeeze())
                encodings = encodings.to(torch.float32)
                weighted = (encodings * pattern_weights).sum(dim=0)
                return to_numpy(weighted.argmax(1)).reshape([-1, 1])

    return EnsemblePattern(patterns, ensemble_method)


class EnsembleResults(NamedTuple):
    data: TabularData
    pipelines: List[Pipeline]
    pattern_weights: Optional[np.ndarray]
    predict_config: Optional[Dict[str, Any]]

    @property
    def pattern(self) -> EnsemblePattern:
        predict_config = self.predict_config or {}
        patterns = [m.to_pattern(**predict_config) for m in self.pipelines]
        return ensemble(patterns, pattern_weights=self.pattern_weights)


class MetricsPlaceholder(NamedTuple):
    config: Dict[str, Any]


class Ensemble:
    def __init__(
        self,
        task_type: task_type_type,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.task_type = task_type
        if config is None:
            config = {}
        self.config = shallow_copy_dict(config)

    def bagging(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        num_jobs: int = 4,
        model: str = "fcnn",
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        identifiers: Optional[Union[str, List[str]]] = None,
        predict_config: Optional[Dict[str, Any]] = None,
        sequential: Optional[bool] = None,
        temp_folder: str = "__tmp__",
        return_patterns: bool = True,
        use_tqdm: bool = True,
    ) -> EnsembleResults:
        repeat_result = repeat_with(
            x,
            y,
            models=model,
            model_configs=model_configs,
            identifiers=identifiers,
            predict_config=predict_config,
            sequential=sequential,
            num_jobs=num_jobs,
            num_repeat=k,
            temp_folder=temp_folder,
            return_patterns=return_patterns,
            use_tqdm=use_tqdm,
            **self.config,
        )

        data = repeat_result.data
        pipelines = repeat_result.pipelines
        assert pipelines is not None
        return EnsembleResults(data, pipelines[model], None, predict_config)

    def adaboost(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        eps: float = 1e-12,
        model: str = "fcnn",
        temp_folder: str = "__tmp__",
        predict_config: Optional[Dict[str, Any]] = None,
        increment_config: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> EnsembleResults:
        if increment_config is None:
            increment_config = {}
        config = shallow_copy_dict(self.config)
        update_dict(increment_config, config)
        config["cv_split"] = 0.0
        config.setdefault("use_tqdm", False)
        config.setdefault("use_binary_threshold", False)
        config.setdefault("verbose_level", 0)

        @register_metric("adaboost_error", -1, False)
        def adaboost_error(
            self_: Union[Metrics, MetricsPlaceholder],
            target_: np.ndarray,
            predictions_: np.ndarray,
        ) -> float:
            target_ = target_.astype(np.float32)
            predictions_ = predictions_.astype(np.float32)
            sample_weights_ = self_.config.get("sample_weights")
            errors = (target_ != predictions_).ravel()
            if sample_weights_ is None:
                e_ = errors.mean()
            else:
                e_ = sample_weights_[errors].sum() / len(errors)
            return e_.item()

        data = None
        pipelines = []
        patterns, pattern_weights = [], []
        for i in tqdm.tqdm(list(range(k))):
            cfg = shallow_copy_dict(config)
            cfg["logging_folder"] = os.path.join(temp_folder, str(i))
            metric_config = {"sample_weights": sample_weights}
            if sample_weights is not None:
                cfg["metrics"] = "adaboost_error"
                cfg["metric_config"] = metric_config
            m = make(model=model, **cfg)
            m.fit(x, y, sample_weights=sample_weights)
            metrics_placeholder = MetricsPlaceholder(metric_config)
            predictions: np.ndarray = m.predict(x, contains_labels=True)
            predictions = predictions.astype(np.float32)
            target = m.data.processed.y.astype(np.float32)
            e = adaboost_error(metrics_placeholder, target, predictions)
            em = min(max(e, eps), 1.0 - eps)
            am = 0.5 * math.log(1.0 / em - 1.0)
            if sample_weights is None:
                sample_weights = np.ones_like(predictions).ravel()
            target[target == 0.0] = predictions[predictions == 0.0] = -1.0
            sample_weights *= np.exp(-am * target * predictions).ravel()
            sample_weights /= np.mean(sample_weights)
            patterns.append(m.to_pattern())
            pattern_weights.append(am)
            if data is None:
                data = m.data
            pipelines.append(m)

        weights_array = np.array(pattern_weights, np.float32)
        weights_array /= weights_array.sum()

        return EnsembleResults(data, pipelines, weights_array, predict_config)


__all__ = [
    "Benchmark",
    "ensemble",
    "Ensemble",
    "EnsembleResults",
]
