import os
import math
import tqdm
import torch

import numpy as np

from typing import *
from cftool.misc import *
from cftool.ml.utils import *
from cfdata.tabular import *
from torch.nn.functional import one_hot

from .zoo import zoo
from .basic import *
from ..dist import *
from ..misc.toolkit import *


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
        task_type: TaskTypes,
        *,
        temp_folder: str = None,
        project_name: str = "carefree-learn",
        models: Union[str, List[str]] = "fcnn",
        increment_config: Dict[str, Any] = None,
        data_config: Dict[str, Any] = None,
        read_config: Dict[str, Any] = None,
        use_tracker: bool = False,
        use_cuda: bool = True,
    ):
        self.data = None
        if data_config is None:
            data_config = {}
        if read_config is None:
            read_config = {}
        self.data_config, self.read_config = data_config, read_config
        self.task_name, self.task_type = task_name, task_type
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
        self.experiments = None

    @property
    def identifier(self) -> str:
        return hash_code(
            f"{self.project_name}{self.task_name}{self.models}{self.increment_config}"
        )

    @property
    def data_tasks(self) -> List[Task]:
        return next(iter(self.experiments.data_tasks.values()))

    def _add_tasks(
        self,
        iterator_name: str,
        data_tasks: List[Task],
        experiments: Experiments,
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> None:
        self.configs = {}
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
        predict_config: Dict[str, Any] = None,
    ) -> BenchmarkResults:
        results = self.experiments.run_tasks(
            num_jobs=num_jobs, run_tasks=run_tasks, load_task=load_task
        )
        comparer_list = []
        for i, data_task in enumerate(self.data_tasks):
            wrappers = {}
            x_te, y_te = data_task.fetch_data("_te")
            for identifier, ms in results.items():
                wrappers[identifier] = ms[i]
            comparer = estimate(
                x_te,
                y_te,
                wrappers=wrappers,
                wrapper_predict_config=predict_config,
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
            self.data, best_configs, best_methods, self.experiments, comparer
        )

    def _pre_process(self, x: data_type, y: data_type = None) -> TabularDataset:
        data_config = shallow_copy_dict(self.data_config)
        task_type = data_config.pop("task_type", None)
        if task_type is not None:
            assert task_type is self.task_type
        self.data = TabularData.simple(self.task_type, **data_config).read(
            x, y, **self.read_config
        )
        return self.data.to_dataset()

    def _k_core(
        self,
        k_iterator: Iterable,
        num_jobs: int,
        run_tasks: bool,
        predict_config: Dict[str, Any],
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]],
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
        predict_config: Dict[str, Any] = None,
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]] = None,
    ) -> BenchmarkResults:
        dataset = self._pre_process(x, y)
        return self._k_core(
            KFold(k, dataset), num_jobs, run_tasks, predict_config, benchmarks
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
        predict_config: Dict[str, Any] = None,
        benchmarks: Dict[str, Dict[str, Dict[str, Any]]] = None,
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
        saving_folder: str,
        *,
        simplify: bool = True,
        compress: bool = True,
    ) -> "Benchmark":
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            Saving.prepare_folder(self, saving_folder)
            Saving.save_dict(
                {
                    "task_name": self.task_name,
                    "task_type": self.task_type.value,
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
            self.experiments.save(
                experiments_folder, simplify=simplify, compress=compress
            )
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        saving_folder: str,
        *,
        predict_config: Dict[str, Any] = None,
        compress: bool = True,
    ) -> Tuple["Benchmark", BenchmarkResults]:
        abs_folder = os.path.abspath(saving_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [saving_folder]):
            with Saving.compress_loader(abs_folder, compress, remove_extracted=False):
                kwargs = Saving.load_dict("kwargs", abs_folder)
                configs = kwargs.pop("configs")
                iterator_name = kwargs.pop("iterator_name")
                kwargs["task_type"] = TaskTypes.from_str(kwargs["task_type"])
                benchmark = cls(**kwargs)
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
    pattern_weights: np.ndarray = None,
    ensemble_method: Union[str, collate_fn_type] = None,
) -> EnsemblePattern:
    if ensemble_method is None:
        if pattern_weights is None:
            ensemble_method = "default"
        else:
            pattern_weights = pattern_weights.reshape([-1, 1, 1])

            def ensemble_method(
                arrays: List[np.ndarray], requires_prob: bool
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
    pattern: EnsemblePattern
    experiments: Union[Experiments, None]


class Ensemble:
    def __init__(self, task_type: TaskTypes, config: Dict[str, Any] = None):
        self.task_type = task_type
        if config is None:
            config = {}
        self.config = config

    def bagging(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        num_test: Union[int, float] = 0.1,
        num_jobs: int = 4,
        run_tasks: bool = True,
        predict_config: Dict[str, Any] = None,
        temp_folder: str = None,
        project_name: str = "carefree-learn",
        task_name: str = "bagging",
        models: Union[str, List[str]] = "fcnn",
        increment_config: Dict[str, Any] = None,
        use_tracker: bool = True,
        use_cuda: bool = True,
    ) -> EnsembleResults:
        if isinstance(models, str):
            models = [models]

        data_config, read_config = map(
            self.config.get, ["data_config", "read_config"], [{}, {}]
        )
        benchmark = Benchmark(
            task_name,
            self.task_type,
            temp_folder=temp_folder,
            project_name=project_name,
            models=models,
            increment_config=increment_config,
            use_tracker=use_tracker,
            use_cuda=use_cuda,
            data_config=data_config,
            read_config=read_config,
        )
        dataset = benchmark._pre_process(x, y)
        k_bootstrap = KBootstrap(k, num_test, dataset)
        benchmark_results = benchmark._k_core(
            k_bootstrap,
            num_jobs,
            run_tasks,
            predict_config,
            {model: {"config": shallow_copy_dict(self.config)} for model in models},
        )

        def _pre_process(x_):
            return benchmark_results.data.transform(x_, contains_labels=False).x

        experiments = benchmark_results.experiments
        ms_dict = transform_experiments(experiments)
        all_models = sum(ms_dict.values(), [])
        all_patterns = [m.to_pattern(pre_process=_pre_process) for m in all_models]
        ensemble_pattern = ensemble(all_patterns)

        return EnsembleResults(benchmark_results.data, ensemble_pattern, experiments)

    def adaboost(
        self,
        x: data_type,
        y: data_type = None,
        *,
        k: int = 10,
        eps: float = 1e-12,
        model: str = "fcnn",
        increment_config: Dict[str, Any] = None,
        sample_weights: Union[np.ndarray, None] = None,
        num_test: Union[int, float] = 0.1,
    ) -> EnsembleResults:
        if increment_config is None:
            increment_config = {}
        config = shallow_copy_dict(self.config)
        update_dict(increment_config, config)
        config["cv_split"] = num_test
        config.setdefault("use_tqdm", False)
        config.setdefault("verbose_level", 0)

        data = None
        patterns, pattern_weights = [], []
        for _ in tqdm.tqdm(list(range(k))):
            m = make(model=model, **config)
            m.fit(x, y, sample_weights=sample_weights)
            predictions = m.predict(x, contains_labels=True).astype(np.float32)
            target = m._original_data.processed.y.astype(np.float32)
            errors = (predictions != target).ravel()
            if sample_weights is None:
                e = errors.mean()
            else:
                e = errors.dot(sample_weights) / len(errors)
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
                data = m._original_data

        pattern_weights = np.array(pattern_weights, np.float32)
        ensemble_pattern = ensemble(patterns, pattern_weights=pattern_weights)
        return EnsembleResults(data, ensemble_pattern, None)


__all__ = [
    "Benchmark",
    "ensemble",
    "Ensemble",
]
