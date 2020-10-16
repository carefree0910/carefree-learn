import os
import json
import torch
import optuna
import logging

import numpy as np

from typing import *
from cftool.ml import Metrics
from cftool.ml import Tracker
from cftool.ml import ModelPattern
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from cftool.misc import update_dict
from cftool.misc import lock_manager
from cftool.misc import timing_context
from cftool.misc import shallow_copy_dict
from trains import Task, Logger
from functools import partial
from cfdata.types import np_int_type

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from ..models.base import model_dict
from ..trainer.core import to_prob
from ..trainer.core import Trainer
from ..misc.toolkit import to_2d
from ..misc.toolkit import data_type


trains_logger: Union[Logger, None] = None


class Pipeline(LoggingMixin):
    def __init__(
        self,
        config: Union[str, Dict[str, Any]] = None,
        *,
        increment_config: Union[str, Dict[str, Any]] = None,
        trial: optuna.trial.Trial = None,
        tracker_config: Dict[str, Any] = None,
        cuda: Union[str, int] = None,
        verbose_level: int = 2,
    ):
        self.trial = trial
        self.tracker = None if tracker_config is None else Tracker(**tracker_config)
        self._verbose_level = int(verbose_level)
        if cuda == "cpu":
            self.device = torch.device("cpu")
        elif cuda is not None:
            self.device = torch.device(f"cuda:{cuda}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config, increment_config = map(
            self._get_config, [config, increment_config]
        )
        update_dict(increment_config, self.config)
        self._init_config()

    def __str__(self) -> str:
        return f"{type(self.model).__name__}()"  # type: ignore

    __repr__ = __str__

    @property
    def train_set(self) -> TabularDataset:
        raw = self.tr_data.raw
        return TabularDataset(*raw.xy, task_type=self.tr_data.task_type)

    @property
    def valid_set(self) -> Optional[TabularDataset]:
        if self.cv_data is None:
            return None
        raw = self.cv_data.raw
        return TabularDataset(*raw.xy, task_type=self.cv_data.task_type)

    @property
    def binary_threshold(self) -> Union[float, None]:
        return self._binary_threshold

    @staticmethod
    def _get_config(config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, str):
            with open(config, "r") as f:
                return json.load(f)
        return shallow_copy_dict(config)

    def _init_config(self) -> None:
        self.timing = self.config.setdefault("use_timing_context", True)
        self._data_config = self.config.setdefault("data_config", {})
        self._data_config["use_timing_context"] = self.timing
        self._data_config["default_categorical_process"] = "identical"
        self._read_config = self.config.setdefault("read_config", {})
        self._cv_split = self.config.setdefault("cv_split", 0.1)
        self._cv_split_order = self.config.setdefault("cv_split_order", "auto")
        self._model = self.config.setdefault("model", "fcnn")
        self._binary_metric = self.config.setdefault("binary_metric", "acc")
        self._is_binary = self.config.get("is_binary")
        self._binary_threshold = self.config.get("binary_threshold")
        self.config.setdefault("use_amp", False)
        logging_folder = self.config["logging_folder"] = self.config.setdefault(
            "logging_folder",
            os.path.join("_logging", model_dict[self._model].__identifier__),
        )
        logging_file = self.config.get("logging_file")
        if logging_file is not None:
            logging_path = os.path.join(logging_folder, logging_file)
        else:
            logging_path = os.path.abspath(self.generate_logging_path(logging_folder))
        self.config["_logging_path_"] = logging_path
        self._init_logging(
            self._verbose_level, self.config.setdefault("trigger_logging", False)
        )

    def _prepare_modules(self, *, is_loading: bool = False) -> None:
        # model
        with timing_context(self, "init model", enable=self.timing):
            self.model = model_dict[self._model](self.config, self.tr_data, self.device)
        # trainer
        with timing_context(self, "init trainer", enable=self.timing):
            self.trainer = Trainer(
                self.model,
                self.trial,
                self.tracker,
                self.config,
                self._verbose_level,
                is_loading,
            )
        # to device
        with timing_context(self, "init device", enable=self.timing):
            self.trainer.to(self.device)

    def _before_loop(
        self,
        x: data_type,
        y: data_type,
        x_cv: data_type,
        y_cv: data_type,
        sample_weights: np.ndarray,
    ) -> None:
        # data
        y, y_cv = map(to_2d, [y, y_cv])
        args = (x, y) if y is not None else (x,)
        self._data_config["verbose_level"] = self._verbose_level
        self._original_data = TabularData(**self._data_config).read(
            *args, **self._read_config
        )
        self.tr_data = self._original_data
        self._save_original_data = False
        self.tr_weights = None
        if x_cv is not None:
            self.cv_data = self.tr_data.copy_to(x_cv, y_cv)
            if sample_weights is not None:
                self.tr_weights = sample_weights[: len(self.tr_data)]
        else:
            if self._cv_split <= 0.0:
                self.cv_data = None
                if sample_weights is not None:
                    self.tr_weights = sample_weights
            else:
                self._save_original_data = True
                split = self.tr_data.split(
                    self._cv_split,
                    order=self._cv_split_order,
                )
                self.cv_data, self.tr_data = split.split, split.remained
                # TODO : utilize cv_weights with sample_weights[split.split_indices]
                if sample_weights is not None:
                    self.tr_weights = sample_weights[split.remained_indices]
        # modules
        self._prepare_modules()

    def _loop(self) -> None:
        # training loop
        self.trainer(self.tr_data, self.cv_data, self.tr_weights)
        # binary threshold
        if self._binary_threshold is None:
            if self.tr_data.num_classes != 2:
                self._is_binary = False
                self._binary_threshold = None
            else:
                self._is_binary = True
                x, y = self.tr_data.raw.x, self.tr_data.processed.y
                probabilities = self.predict_prob(x)
                try:
                    threshold = Metrics.get_binary_threshold(
                        y, probabilities, self._binary_metric
                    )
                    self._binary_threshold = threshold
                except ValueError:
                    self._binary_threshold = None
        # logging
        self.log_timing()

    # api

    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        sample_weights: np.ndarray = None,
    ) -> "Pipeline":
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self._loop()
        return self

    def trains(
        self,
        x: data_type,
        y: data_type = None,
        x_cv: data_type = None,
        y_cv: data_type = None,
        *,
        sample_weights: np.ndarray = None,
        trains_config: Dict[str, Any] = None,
        keep_task_open: bool = False,
        queue: str = None,
    ) -> "Pipeline":
        if trains_config is None:
            return self.fit(x, y, x_cv, y_cv, sample_weights=sample_weights)
        # init trains
        if trains_config is None:
            trains_config = {}
        project_name = trains_config.get("project_name")
        task_name = trains_config.get("task_name")
        if queue is None:
            task = Task.init(**trains_config)
            cloned_task = None
        else:
            task = Task.get_task(project_name=project_name, task_name=task_name)
            cloned_task = Task.clone(source_task=task, parent=task.id)
        # before loop
        self._verbose_level = 6
        self._data_config["verbose_level"] = 6
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self.trainer.use_tqdm = False
        copied_config = shallow_copy_dict(self.config)
        if queue is not None:
            assert cloned_task is not None
            cloned_task.set_parameters(copied_config)
            Task.enqueue(cloned_task.id, queue)
            return self
        # loop
        task.connect(copied_config)
        global trains_logger
        trains_logger = task.get_logger()
        self._loop()
        if not keep_task_open:
            task.close()
            trains_logger = None
        return self

    def predict(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        requires_recover: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.tr_data.is_reg:
            predictions = self.trainer.predict(x, return_all, **kwargs)
            recover = partial(self.tr_data.recover_labels, inplace=True)
            if not return_all:
                if requires_recover:
                    return recover(predictions)
                return predictions
            if not requires_recover:
                return predictions
            return {k: recover(v) for k, v in predictions.items()}
        probabilities = self.predict_prob(x, **kwargs)
        if not self._is_binary or self._binary_threshold is None:
            return probabilities.argmax(1).reshape([-1, 1])
        return (
            (probabilities[..., 1] >= self._binary_threshold)
            .astype(np_int_type)
            .reshape([-1, 1])
        )

    def predict_prob(self, x: data_type, **kwargs: Any) -> np.ndarray:
        if self.tr_data.is_reg:
            raise ValueError("`predict_prob` should not be called on regression tasks")
        if kwargs and self.trainer.onnx is not None:
            self.log_msg(
                "`kwargs` is provided but onnx is in use, it will be ignored",
                self.warning_prefix,
                msg_level=logging.WARNING,
            )
        raw = self.trainer.predict(x, **kwargs)
        return to_prob(raw)

    def to_pattern(
        self,
        *,
        pre_process: Callable = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        def _predict(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            return self.predict(x, **predict_kwargs)

        def _predict_prob(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            return self.predict_prob(x, **predict_kwargs)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    def save(self, folder: str = None, *, compress: bool = True) -> "Pipeline":
        if folder is None:
            folder = self.trainer.checkpoint_folder
        abs_folder = os.path.abspath(folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [folder]):
            Saving.prepare_folder(self, folder)
            train_data_folder = os.path.join(folder, "__data__", "train")
            if self._save_original_data:
                self._original_data.save(train_data_folder, compress=compress)
            else:
                self.tr_data.save(train_data_folder, compress=compress)
                if self.cv_data is not None:
                    self.cv_data.save(
                        os.path.join(folder, "__data__", "valid"), compress=compress
                    )
            self.trainer.save_checkpoint(folder)
            self.config["is_binary"] = self._is_binary
            self.config["binary_threshold"] = self._binary_threshold
            Saving.save_dict(self.config, "config", folder)
            if compress:
                Saving.compress(abs_folder, remove_original=True)
        return self

    @classmethod
    def load(
        cls,
        folder: str,
        *,
        cuda: int = None,
        verbose_level: int = 0,
        compress: bool = True,
    ) -> "Pipeline":
        base_folder = os.path.dirname(os.path.abspath(folder))
        with lock_manager(base_folder, [folder]):
            with Saving.compress_loader(folder, compress, remove_extracted=True):
                config = Saving.load_dict("config", folder)
                pipeline = Pipeline(config, cuda=cuda, verbose_level=verbose_level)
                tr_data_folder = os.path.join(folder, "__data__", "train")
                cv_data_folder = os.path.join(folder, "__data__", "valid")
                tr_data = pipeline.tr_data = TabularData.load(
                    tr_data_folder, compress=compress
                )
                cv_data = None
                if os.path.isdir(cv_data_folder) or os.path.isfile(
                    f"{cv_data_folder}.zip"
                ):
                    cv_data = pipeline.cv_data = TabularData.load(
                        cv_data_folder, compress=compress
                    )
                pipeline._prepare_modules(is_loading=True)
                trainer = pipeline.trainer
                trainer.restore_checkpoint(folder)
                trainer._init_data(tr_data, cv_data)
                trainer._init_metrics()
        return pipeline


__all__ = ["Pipeline"]
