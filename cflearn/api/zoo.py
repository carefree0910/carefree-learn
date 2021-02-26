import os
import json
import logging

import numpy as np

from abc import *
from typing import *

from cfdata.tabular import KRandom
from cfdata.tabular import TaskTypes
from cfdata.tabular import TabularDataset
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cftool.misc import LoggingMixin

from .basic import make
from ..dist import Experiment
from ..types import data_type
from ..pipeline import Pipeline
from ..protocol import ModelProtocol

registered_benchmarks: Dict[str, Dict[str, Dict[str, Any]]] = {}


class ZooSearchResult(NamedTuple):
    best_key: str
    mapping: Dict[str, str]
    configs: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Dict[str, float]]

    @property
    def best_model(self) -> str:
        return self.mapping[self.best_key]

    @property
    def best_config(self) -> Dict[str, Any]:
        return self.configs[self.best_key]


class Zoo(LoggingMixin, metaclass=ABCMeta):
    def __init__(
        self,
        model: str,
        *,
        model_type: str = "default",
        increment_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_type = model_type
        self.increment_config = increment_config

    @property
    def benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        this method should return a dict of configs (which represent benchmarks)
        * Note that "default" key should always be included in the returned dict
        """
        return registered_benchmarks[self.model]

    @property
    def config(self) -> dict:
        """
        return corresponding config of `self.model_type`
        * and update with `increment_config` if provided
        """
        benchmarks = self.benchmarks
        assert "default" in benchmarks, "'default' should be included in config_dict"
        config = benchmarks.get(self.model_type)
        if config is None:
            if self.model_type != "default":
                self.log_msg(
                    f"model_type '{self.model_type}' is not recognized, "
                    "'default' model_type will be used",
                    self.warning_prefix,
                    2,
                    msg_level=logging.WARNING,
                )
                self.model_type = "default"
            config = self.benchmarks["default"]
        new_config = shallow_copy_dict(config)
        if self.increment_config is not None:
            update_dict(self.increment_config, new_config)
        return new_config

    @property
    def m(self) -> Pipeline:
        """ return corresponding model of self.config """
        return make(self.model, **self.config)

    @classmethod
    def search(
        cls,
        task_type: TaskTypes,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        models: Union[str, List[str]] = "fcnn",
        *,
        workplace: str = "__zoo_search__",
        num_jobs: int = 0,
        num_repeat: int = 5,
        compress: bool = True,
        use_tqdm: bool = True,
        cv_split: Union[int, float] = 0.1,
        increment_config: Optional[Dict[str, Any]] = None,
    ) -> ZooSearchResult:
        if isinstance(models, str):
            models = [models]
        num_jobs = max(1, num_jobs)
        experiment = Experiment(num_jobs=num_jobs)
        model_mapping: Dict[str, str] = {}
        model_configs: Dict[str, Dict[str, Any]] = {}
        data_folder = None
        data_folders = None
        if x_cv is not None:
            args = x, y, x_cv, y_cv
            data_folder = Experiment.dump_data_bundle(*args, workplace=workplace)
        else:
            data_folders = []
            dataset = TabularDataset(x, y, task_type)
            k_random = KRandom(num_repeat, cv_split, dataset)
            for i, (train_split, test_split) in enumerate(k_random):
                train_dataset = train_split.dataset
                test_dataset = test_split.dataset
                x_tr, y_tr = train_dataset.xy
                x_te, y_te = test_dataset.xy
                local_data_folder = os.path.join(workplace, "__data__", str(i))
                os.makedirs(local_data_folder, exist_ok=True)
                experiment.dump_data(local_data_folder, x_tr, y_tr)
                experiment.dump_data(local_data_folder, x_te, y_te, "_te")
                data_folders.append(local_data_folder)
        for i in range(num_repeat):
            if data_folder is not None:
                i_data_folder = data_folder
            else:
                i_data_folder = data_folders[i]
            for model in models:
                for key, config in cls(model).benchmarks.items():
                    config = shallow_copy_dict(config)
                    if increment_config is not None:
                        update_dict(increment_config, config)
                    model_key = f"{model}_{key}"
                    model_mapping[model_key] = model
                    model_configs[model_key] = shallow_copy_dict(config)
                    experiment.add_task(
                        model=model,
                        compress=compress,
                        root_workplace=workplace,
                        workplace_key=(model_key, str(i)),
                        config=shallow_copy_dict(config),
                        data_folder=i_data_folder,
                    )
        results = experiment.run_tasks(use_tqdm=use_tqdm)
        model_keys: List[str] = []
        model_means: List[float] = []
        model_statistics: List[Dict[str, float]] = []
        for model_key, checkpoint_folders in results.checkpoint_folders.items():
            scores = []
            for folder in checkpoint_folders:
                with open(os.path.join(folder, ModelProtocol.scores_file), "r") as f:
                    scores.append(max(json.load(f).values()))
            scores_np = np.array(scores)
            mean, std = scores_np.mean().item(), scores_np.std().item()
            model_statistics.append({"mean": mean, "std": std})
            model_means.append(mean)
            model_keys.append(model_key)
        best_idx = np.argmax(model_means).item()
        best_key = model_keys[best_idx]
        result = ZooSearchResult(
            best_key,
            model_mapping,
            model_configs,
            dict(zip(model_keys, model_statistics)),
        )
        with open(os.path.join(workplace, "result.json"), "w") as f:
            json.dump(result._asdict(), f)
        return result

    @classmethod
    def register(
        cls,
        model: str,
        model_type: str,
        *,
        transform_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        extractor_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        head_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        increment_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        global registered_benchmarks
        model_dict = registered_benchmarks.setdefault(model, {})
        pipe_configs: Dict[str, Any] = {}
        if transform_configs is not None:
            for pipe, transform_config in transform_configs.items():
                pipe_configs.setdefault(pipe, {})["transform"] = transform_config
        if extractor_configs is not None:
            for pipe, extractor_config in extractor_configs.items():
                pipe_configs.setdefault(pipe, {})["extractor"] = extractor_config
        if head_configs is not None:
            for pipe, head_config in head_configs.items():
                pipe_configs.setdefault(pipe, {})["head"] = head_config
        config = {}
        if pipe_configs:
            config = {"model_config": {"pipe_configs": pipe_configs}}
        if increment_configs is not None:
            update_dict(increment_configs, config)
        model_dict[model_type] = config


# fcnn

Zoo.register("fcnn", "default")
Zoo.register(
    "fcnn",
    "min_max",
    increment_configs={"data_config": {"default_numerical_process": "min_max"}},
)
Zoo.register(
    "fcnn",
    "on_large",
    head_configs={"fcnn": {"mapping_configs": {"dropout": 0.1, "batch_norm": False}}},
)
Zoo.register(
    "fcnn",
    "on_log_large",
    head_configs={"fcnn": {"mapping_configs": {"dropout": 0.1, "batch_norm": False}}},
    increment_configs={"data_config": {"default_numerical_process": "logarithm"}},
)
Zoo.register(
    "fcnn",
    "light",
    head_configs={"fcnn": {"mapping_configs": {"batch_norm": False}}},
    increment_configs={
        "data_config": {"binning_method": "opt"},
        "model_config": {"default_encoding_configs": {"embedding_dim": 8}},
    },
)

# tree dnn

Zoo.register("tree_dnn", "default")
Zoo.register(
    "tree_dnn",
    "min_max",
    increment_configs={"data_config": {"default_numerical_process": "min_max"}},
)
Zoo.register(
    "tree_dnn",
    "on_large",
    head_configs={"fcnn": {"mapping_configs": {"dropout": 0.1, "batch_norm": False}}},
)
Zoo.register(
    "tree_dnn",
    "on_log_large",
    head_configs={"fcnn": {"mapping_configs": {"dropout": 0.1, "batch_norm": False}}},
    increment_configs={"data_config": {"default_numerical_process": "logarithm"}},
)
Zoo.register(
    "tree_dnn",
    "light",
    head_configs={
        "dndf": {"dndf_config": None},
        "fcnn": {"mapping_configs": {"batch_norm": False}},
    },
    increment_configs={
        "data_config": {"binning_method": "opt"},
        "model_config": {"default_encoding_configs": {"embedding_dim": 8}},
    },
)


__all__ = ["Zoo"]
