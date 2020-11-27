import os
import torch

import numpy as np

from typing import *
from cfdata.tabular import DataLoader
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset
from cftool.ml import ModelPattern
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import timing_context
from cftool.misc import Saving
from cftool.misc import LoggingMixin

try:
    amp: Optional[Any] = torch.cuda.amp
except:
    amp = None

from .types import data_type
from .configs import Elements
from .configs import Environment
from .trainer import Trainer
from .inference import Inference
from .inference import PreProcessor
from .misc.toolkit import to_2d
from .misc.toolkit import PrefetchLoader
from .misc.time_series import TSLabelCollator
from .models.base import model_dict
from .models.base import ModelBase


class Pipeline(LoggingMixin):
    def __init__(self, environment: Environment):
        self.environment = environment
        self.device = environment.device
        self.model: Optional[ModelBase] = None
        self.inference: Optional[Inference]
        LoggingMixin.reset_logging()
        self.config = environment.pipeline_config
        self.model_type = environment.model
        self.timing = self.config.setdefault("use_timing_context", True)
        self.data_config["use_timing_context"] = self.timing
        self.data_config["default_categorical_process"] = "identical"
        self.sampler_config = self.config.setdefault("sampler_config", {})
        self._init_logging(environment.verbose_level, environment.trigger_logging)

    def __getattr__(self, item: str) -> Any:
        return self.environment.config.get(item)

    def __str__(self) -> str:
        return f"{type(self.model).__name__}()"  # type: ignore

    __repr__ = __str__

    @property
    def data(self) -> TabularData:
        return self._original_data

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
    def int_cv_split(self) -> int:
        if isinstance(self.cv_split, int):
            return self.cv_split
        num_data = len(self._original_data)
        if self.cv_split is not None:
            return int(round(self.cv_split * num_data))
        default_cv_split = 0.1
        cv_split_num = int(round(default_cv_split * num_data))
        cv_split_num = max(self.min_cv_split, cv_split_num)
        max_cv_split = int(round(num_data * self.max_cv_split_ratio))
        max_cv_split = min(self.max_cv_split, max_cv_split)
        return min(cv_split_num, max_cv_split)

    @property
    def binary_threshold(self) -> Optional[float]:
        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        return self.inference.binary_threshold

    def _init_data(self) -> None:
        if not self.data.is_ts:
            self.ts_label_collator = None
        else:
            self.ts_label_collator = TSLabelCollator(
                self.data,
                self.ts_label_collator_config,
            )
        self.sampler_config.setdefault("verbose_level", self.data._verbose_level)
        self.preprocessor = PreProcessor(self._original_data, self.sampler_config)
        tr_sampler = self.preprocessor.make_sampler(
            self.tr_data,
            self.shuffle_tr,
            self.tr_weights,
        )
        self.tr_loader = DataLoader(
            self.batch_size,
            tr_sampler,
            return_indices=True,
            verbose_level=self._verbose_level,
            label_collator=self.ts_label_collator,
        )
        if self.cv_data is None:
            self.cv_loader = None
        else:
            cv_sampler = self.preprocessor.make_sampler(self.cv_data, False)
            self.cv_loader = DataLoader(
                self.cv_batch_size,
                cv_sampler,
                return_indices=True,
                verbose_level=self._verbose_level,
                label_collator=self.ts_label_collator,
            )
            self.cv_loader.enabled_sampling = False
        # tr loader copy
        self.tr_loader_copy = self.tr_loader.copy()
        self.tr_loader_copy.enabled_sampling = False
        self.tr_loader_copy.sampler.shuffle = False

    def _prepare_modules(self, *, is_loading: bool = False) -> None:
        # model
        with timing_context(self, "init model", enable=self.timing):
            self.model = model_dict[self.model_type](
                self.environment,
                self.tr_loader_copy,
                self.cv_loader,
                self.tr_weights,
                self.cv_weights,
            )
            self.model.init_ema()
        # trainer
        with timing_context(self, "init trainer", enable=self.timing):
            if self.preprocessor is None:
                msg = "`preprocessor` is not defined. Please call `_init_data` first"
                raise ValueError(msg)
            self.inference = Inference(
                self.preprocessor,
                model=self.model,
                binary_config=self.binary_config,
                use_binary_threshold=self.use_binary_threshold,
                use_tqdm=self.use_tqdm,
            )
            self.trainer = Trainer(
                self.model,
                self.inference,
                self.environment,
                is_loading,
            )
        # to device
        with timing_context(self, "init device", enable=self.timing):
            self.trainer.model.to(self.device)

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
        self.data_config["verbose_level"] = self._verbose_level
        if sample_weights is None:
            self.sample_weights = None
        else:
            self.sample_weights = sample_weights.copy()
        self._original_data = TabularData(**self.data_config)
        self._original_data.read(*args, **self.read_config)
        self.tr_data = self._original_data
        self._save_original_data = x_cv is None
        self.tr_weights = self.cv_weights = None
        if x_cv is not None:
            self.cv_data = self.tr_data.copy_to(x_cv, y_cv)
            if sample_weights is not None:
                self.tr_weights = sample_weights[: len(self.tr_data)]
                self.cv_weights = sample_weights[len(self.tr_data) :]
        else:
            if self.int_cv_split <= 0:
                self.cv_data = None
                self.tr_split_indices = None
                self.cv_split_indices = None
                if sample_weights is not None:
                    self.tr_weights = sample_weights
            else:
                split = self.tr_data.split(self.int_cv_split, order=self.cv_split_order)
                self.tr_data, self.cv_data = split.remained, split.split
                self.tr_split_indices = split.remained_indices
                self.cv_split_indices = split.split_indices
                # TODO : utilize cv_weights with sample_weights[split.split_indices]
                if sample_weights is not None:
                    self.tr_weights = sample_weights[split.remained_indices]
                    self.cv_weights = sample_weights[split.split_indices]
        self._init_data()
        # modules
        self._prepare_modules()

    def _loop(self) -> None:
        # dump information
        logging_folder = self.logging_folder
        os.makedirs(logging_folder, exist_ok=True)
        Saving.save_dict(self.config, "config", logging_folder)
        with open(os.path.join(logging_folder, "model.txt"), "w") as f:
            f.write(str(self.model))
        # training loop
        self.trainer.fit(
            self.tr_loader,
            self.tr_loader_copy,
            self.cv_loader,
            self.tr_weights,
            self.cv_weights,
        )
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
        sample_weights: Optional[np.ndarray] = None,
    ) -> "Pipeline":
        self._before_loop(x, y, x_cv, y_cv, sample_weights)
        self._loop()
        return self

    def predict(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        contains_labels: bool = False,
        requires_recover: bool = True,
        returns_probabilities: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        loader = self.preprocessor.make_inference_loader(
            x,
            self.device,
            self.cv_batch_size,
            is_onnx=self.inference.onnx is not None,
            contains_labels=contains_labels,
        )
        kwargs = shallow_copy_dict(kwargs)
        kwargs.update(
            {
                "return_all": return_all,
                "requires_recover": requires_recover,
                "returns_probabilities": returns_probabilities,
            }
        )

        if self.inference is None:
            raise ValueError("`inference` is not yet generated")
        return self.inference.predict(loader, **shallow_copy_dict(kwargs))

    def predict_prob(
        self,
        x: data_type,
        *,
        return_all: bool = False,
        contains_labels: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.data.is_reg:
            raise ValueError("`predict_prob` should not be called on regression tasks")
        return self.predict(
            x,
            return_all=return_all,
            contains_labels=contains_labels,
            returns_probabilities=True,
            **shallow_copy_dict(kwargs),
        )

    def to_pattern(
        self,
        *,
        pre_process: Optional[Callable] = None,
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

    data_folder = "data"
    train_folder = "train"
    valid_folder = "valid"
    original_folder = "original"
    train_indices_file = "train_indices.npy"
    valid_indices_file = "valid_indices.npy"
    sample_weights_file = "sample_weights.npy"

    @classmethod
    def make(cls, config: Dict[str, Any]) -> "Pipeline":
        return cls(Environment.from_elements(Elements.make(config)))

    def save(
        self,
        export_folder: Optional[str] = None,
        *,
        compress: bool = True,
        retain_data: bool = True,
        remove_original: bool = True,
    ) -> "Pipeline":
        if export_folder is None:
            export_folder = self.trainer.checkpoint_folder
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            Saving.prepare_folder(self, export_folder)
            data_folder = os.path.join(export_folder, self.data_folder)
            os.makedirs(data_folder, exist_ok=True)
            if self.sample_weights is not None:
                sw_file = os.path.join(data_folder, self.sample_weights_file)
                np.save(sw_file, self.sample_weights)
            if not self._save_original_data:
                train_data_folder = os.path.join(data_folder, self.train_folder)
                valid_data_folder = os.path.join(data_folder, self.valid_folder)
                self.tr_data.save(
                    train_data_folder,
                    retain_data=retain_data,
                    compress=False,
                )
                self.cv_data.save(
                    valid_data_folder,
                    retain_data=retain_data,
                    compress=False,
                )
            else:
                original_data_folder = os.path.join(data_folder, self.original_folder)
                self._original_data.save(
                    original_data_folder,
                    retain_data=retain_data,
                    compress=False,
                )
                if self.tr_split_indices is not None:
                    tr_file = os.path.join(data_folder, self.train_indices_file)
                    np.save(tr_file, self.tr_split_indices)
                if self.cv_split_indices is not None:
                    cv_file = os.path.join(data_folder, self.valid_indices_file)
                    np.save(cv_file, self.cv_split_indices)
            final_results = self.trainer.final_results
            if final_results is None:
                raise ValueError("`final_results` are not generated yet")
            score = final_results.final_score
            self.trainer.save_checkpoint(score, export_folder)
            if self.inference is None:
                raise ValueError("`inference` is not yet generated")
            self.config["binary_config"] = self.inference.binary_config
            Saving.save_dict(self.config, "config", export_folder)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def load(
        cls,
        export_folder: str,
        *,
        compress: bool = True,
        verbose_level: int = 0,
        cuda: Optional[Union[int, str]] = None,
    ) -> "Pipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                config = Saving.load_dict("config", export_folder)
                config.update({"verbose_level": verbose_level, "cuda": cuda})
                pipeline = cls.make(config)
                data_folder = os.path.join(export_folder, cls.data_folder)
                # sample weights
                tr_weights = cv_weights = sample_weights = None
                sw_file = os.path.join(data_folder, cls.sample_weights_file)
                if os.path.isfile(sw_file):
                    sample_weights = np.load(sw_file)
                # data
                original_data_folder = os.path.join(data_folder, cls.original_folder)
                if not os.path.isdir(original_data_folder):
                    train_data_folder = os.path.join(data_folder, cls.train_folder)
                    valid_data_folder = os.path.join(data_folder, cls.valid_folder)
                    try:
                        tr_data = TabularData.load(train_data_folder, compress=False)
                        cv_data = TabularData.load(valid_data_folder, compress=False)
                    except Exception as e:
                        raise ValueError(
                            f"data information is corrupted ({e}), "
                            "this may cause by backward compatible breaking"
                        )
                    original_data = tr_data
                    if sample_weights is not None:
                        tr_weights = sample_weights[: len(tr_data)]
                        cv_weights = sample_weights[len(tr_data) :]
                else:
                    original_data = TabularData.load(
                        original_data_folder,
                        compress=False,
                    )
                    vi_file = os.path.join(data_folder, cls.valid_indices_file)
                    if not os.path.isfile(vi_file):
                        tr_weights = sample_weights
                        tr_data = original_data
                        cv_data = None
                    else:
                        ti_file = os.path.join(data_folder, cls.train_indices_file)
                        train_indices, valid_indices = map(np.load, [ti_file, vi_file])
                        split = original_data.split_with_indices(
                            valid_indices, train_indices
                        )
                        tr_data, cv_data = split.remained, split.split
                        if sample_weights is not None:
                            tr_weights = sample_weights[train_indices]
                            cv_weights = sample_weights[valid_indices]
                pipeline.sample_weights = sample_weights
                pipeline.tr_weights = tr_weights
                pipeline.cv_weights = cv_weights
                pipeline._original_data = original_data
                pipeline.tr_data = tr_data
                pipeline.cv_data = cv_data
                pipeline._init_data()
                pipeline._prepare_modules(is_loading=True)
                trainer = pipeline.trainer
                trainer.tr_loader = PrefetchLoader(pipeline.tr_loader, pipeline.device)
                cv_loader = pipeline.cv_loader
                if cv_loader is None:
                    trainer.cv_loader = None
                else:
                    trainer.cv_loader = PrefetchLoader(cv_loader, pipeline.device)
                trainer.restore_checkpoint(export_folder)
                trainer._init_metrics()
        return pipeline


__all__ = ["Pipeline"]
