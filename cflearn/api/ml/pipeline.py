import os
import copy
import json
import torch

import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from cfdata.tabular import ColumnTypes
from cfdata.tabular import TabularData
from cftool.ml import ModelPattern
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving

from ...types import data_type
from ...trainer import get_sorted_checkpoints
from ...trainer import Trainer
from ...trainer import DeviceInfo
from ...protocol import loss_dict
from ...protocol import LossProtocol
from ...protocol import MetricProtocol
from ...protocol import InferenceOutputs
from ...constants import SCORES_FILE
from ...constants import INFO_PREFIX
from ...constants import WARNING_PREFIX
from ...constants import PREDICTIONS_KEY
from ..internal_.trainer import make_trainer
from ...misc.toolkit import get_arguments
from ...misc.toolkit import prepare_workplace_from
from ...misc.internal_ import MLData
from ...misc.internal_ import MLLoader
from ...misc.internal_ import MLInference
from ...models.ml.encoders import Encoder
from ...models.ml.protocol import MLModel


class MLPipeline:
    data: TabularData
    loss: LossProtocol
    model: MLModel
    trainer: Trainer
    inference: MLInference
    device_info: DeviceInfo

    train_data: TabularData
    valid_data: Optional[TabularData]
    train_loader: MLLoader
    train_loader_copy: MLLoader
    valid_loader: MLLoader
    encoder: Optional[Encoder]

    configs_file: str = "configs.json"
    metrics_log_file: str = "metrics.txt"

    data_folder: str = "data"
    final_results_file = "final_results.json"
    config_bundle_name = "config_bundle"

    def __init__(
        self,
        core_name: str = "fcnn",
        core_config: Optional[Dict[str, Any]] = None,
        *,
        loss_name: str = "auto",
        loss_config: Optional[Dict[str, Any]] = None,
        # data
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        # valid split
        valid_split: Optional[Union[int, float]] = None,
        min_valid_split: int = 100,
        max_valid_split: int = 10000,
        max_valid_split_ratio: float = 0.5,
        valid_split_order: str = "auto",
        # data loader
        num_history: int = 1,
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
        # encoder
        only_categorical: bool = False,
        encoder_config: Optional[Dict[str, Any]] = None,
        encoding_methods: Optional[Dict[str, List[str]]] = None,
        encoding_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_encoding_methods: Optional[List[str]] = None,
        default_encoding_configs: Optional[Dict[str, Any]] = None,
        # trainer
        state_config: Optional[Dict[str, Any]] = None,
        num_epoch: int = 40,
        valid_portion: float = 1.0,
        amp: bool = False,
        clip_norm: float = 0.0,
        metric_names: Optional[Union[str, List[str]]] = None,
        metric_configs: Optional[Dict[str, Any]] = None,
        monitor_names: Optional[Union[str, List[str]]] = None,
        monitor_configs: Optional[Dict[str, Any]] = None,
        callback_names: Optional[Union[str, List[str]]] = None,
        callback_configs: Optional[Dict[str, Any]] = None,
        optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        workplace: str = "_logs",
        rank: Optional[int] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
        # misc
        in_loading: bool = False,
    ):
        self.config = get_arguments()
        self.config.pop("self")
        self.core_name = core_name
        self.core_config = core_config or {}
        self.loss_name = loss_name
        self.loss_config = loss_config or {}
        if data_config is None:
            data_config = {}
        data_config["default_categorical_process"] = "identical"
        self.data = TabularData(**(data_config or {}))
        self.processed_dim: Optional[int] = None
        self.read_config = read_config or {}
        self.valid_split = valid_split
        self.min_valid_split = min_valid_split
        self.max_valid_split = max_valid_split
        self.max_valid_split_ratio = max_valid_split_ratio
        self.valid_split_order = valid_split_order
        self.num_history = num_history
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.only_categorical = only_categorical
        self.encoder_config = encoder_config or {}
        self.encoding_methods = encoding_methods or {}
        self.encoding_configs = encoding_configs or {}
        if default_encoding_methods is None:
            default_encoding_methods = ["embedding"]
        self.default_encoding_methods = default_encoding_methods
        self.default_encoding_configs = default_encoding_configs or {}
        self.trainer_config: Dict[str, Any] = {
            "state_config": state_config,
            "num_epoch": num_epoch,
            "valid_portion": valid_portion,
            "amp": amp,
            "clip_norm": clip_norm,
            "metric_names": metric_names,
            "metric_configs": metric_configs,
            "monitor_names": monitor_names,
            "monitor_configs": monitor_configs,
            "callback_names": callback_names,
            "callback_configs": callback_configs,
            "optimizer_settings": optimizer_settings,
            "workplace": workplace,
            "rank": rank,
            "tqdm_settings": tqdm_settings,
        }
        self.in_loading = in_loading

    @property
    def device(self) -> torch.device:
        return self.device_info.device

    def _prepare_data(self) -> None:
        train_loader = MLLoader(
            MLData(*self.train_data.processed.xy),
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
        )
        train_loader_copy = copy.deepcopy(train_loader)
        train_loader_copy.shuffle = False
        if self.valid_data is None:
            valid_loader = train_loader_copy
        else:
            valid_loader = MLLoader(
                MLData(*self.valid_data.processed.xy),
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
            )
        self.train_loader = train_loader
        self.train_loader_copy = train_loader_copy
        self.valid_loader = valid_loader

    def _prepare_modules(self) -> None:
        if not self.in_loading:
            workplace = prepare_workplace_from(self.trainer_config["workplace"])
            self.trainer_config["workplace"] = workplace
            self.trainer_config["metrics_log_file"] = self.metrics_log_file
            os.makedirs(workplace, exist_ok=True)
            with open(os.path.join(workplace, self.configs_file), "w") as f:
                json.dump(self.config, f)
        # encoder
        excluded = 0
        numerical_columns_mapping = {}
        categorical_columns_mapping = {}
        categorical_dims = []
        encoding_methods = []
        encoding_configs = []
        true_categorical_columns = []
        use_one_hot = False
        use_embedding = False
        if self.data.is_simplify:
            for idx in range(self.data.processed.x.shape[1]):
                numerical_columns_mapping[idx] = idx
        else:
            ts_indices = self.data.ts_indices
            recognizers = self.data.recognizers
            sorted_indices = [idx for idx in sorted(recognizers) if idx != -1]
            for idx in sorted_indices:
                recognizer = recognizers[idx]
                assert recognizer is not None
                if not recognizer.info.is_valid or idx in ts_indices:
                    excluded += 1
                elif recognizer.info.column_type is ColumnTypes.NUMERICAL:
                    numerical_columns_mapping[idx] = idx - excluded
                else:
                    str_idx = str(idx)
                    categorical_dims.append(recognizer.num_unique_values)
                    idx_encoding_methods = self.encoding_methods.setdefault(
                        str_idx,
                        self.default_encoding_methods,
                    )
                    if isinstance(idx_encoding_methods, str):
                        idx_encoding_methods = [idx_encoding_methods]
                    use_one_hot = use_one_hot or "one_hot" in idx_encoding_methods
                    use_embedding = use_embedding or "embedding" in idx_encoding_methods
                    encoding_methods.append(idx_encoding_methods)
                    encoding_configs.append(
                        self.encoding_configs.setdefault(
                            str_idx,
                            self.default_encoding_configs,
                        )
                    )
                    true_idx = idx - excluded
                    true_categorical_columns.append(true_idx)
                    categorical_columns_mapping[idx] = true_idx
        if not true_categorical_columns:
            encoder = None
        else:
            if self.in_loading:
                loaders = []
            else:
                loaders = [self.train_loader_copy]
                if self.valid_loader is not None:
                    loaders.append(self.valid_loader)
            encoder = Encoder(
                self.encoder_config,
                categorical_dims,
                encoding_methods,  # type: ignore
                encoding_configs,
                true_categorical_columns,
                loaders,
            )
        self.encoder = encoder
        # prepare
        if self.loss_name == "auto":
            self.loss_name = "focal" if self.data.is_clf else "mae"
        self.loss = loss_dict[self.loss_name](**(self.loss_config or {}))
        if self.processed_dim is None:
            self.processed_dim = self.data.processed_dim
        self.model = MLModel(
            self.processed_dim,
            1 if self.data.is_reg else self.data.num_classes,
            self.num_history,
            encoder=encoder,
            numerical_columns_mapping=numerical_columns_mapping,
            categorical_columns_mapping=categorical_columns_mapping,
            use_one_hot=use_one_hot,
            use_embedding=use_embedding,
            only_categorical=self.only_categorical,
            core_name=self.core_name,
            core_config=self.core_config,
        )
        self.inference = MLInference(self.model)

    def _prepare_trainer_defaults(self) -> None:
        # set some trainer defaults to ml tasks which work well in practice
        if self.trainer_config["metric_names"] is None and self.data.is_clf:
            self.trainer_config["metric_names"] = ["acc", "auc"]
        if self.trainer_config["monitor_names"] is None:
            self.trainer_config["monitor_names"] = ["mean_std", "plateau"]
        auto_callback_setup = False
        tqdm_settings = self.trainer_config["tqdm_settings"]
        callback_names = self.trainer_config["callback_names"]
        optimizer_settings = self.trainer_config["optimizer_settings"]
        if callback_names is None:
            callback_names = []
            auto_callback_setup = True
        if isinstance(callback_names, str):
            callback_names = [callback_names]
        if "log_metrics_msg" not in callback_names and auto_callback_setup:
            if tqdm_settings is None or not tqdm_settings.get("use_tqdm", False):
                callback_names.append("log_metrics_msg")
        if "_default_opt_settings" not in callback_names:
            callback_names.append("_default_opt_settings")
        if "_inject_loader_name" not in callback_names:
            callback_names.append("_inject_loader_name")
        if optimizer_settings is None:
            optimizer_settings = {"all": {"optimizer": "adam", "scheduler": "warmup"}}
        self.trainer_config["tqdm_settings"] = tqdm_settings
        self.trainer_config["callback_names"] = callback_names
        self.trainer_config["optimizer_settings"] = optimizer_settings

    # TODO : support sample weights
    def fit(
        self,
        x: data_type,
        y: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        cuda: Optional[str] = None,
    ) -> "MLPipeline":
        # prepare data
        self.data.read(x, y, **self.read_config)
        if x_valid is not None:
            self.train_data = self.data
            self.valid_data = self.data.copy_to(x_valid, y_valid)
        else:
            if isinstance(self.valid_split, int):
                split = self.valid_split
            else:
                num_data = len(self.data)
                if isinstance(self.valid_split, float):
                    split = int(round(self.valid_split * num_data))
                else:
                    default_split = 0.1
                    num_split = int(round(default_split * num_data))
                    num_split = max(self.min_valid_split, num_split)
                    max_split = int(round(num_data * self.max_valid_split_ratio))
                    max_split = min(max_split, self.max_valid_split)
                    split = min(num_split, max_split)
            if split <= 0:
                self.train_data = self.data
                self.valid_data = None
            else:
                split_result = self.data.split(split, order=self.valid_split_order)
                self.train_data = split_result.remained
                self.valid_data = split_result.split
        # internal preparation
        self._prepare_data()
        self._prepare_modules()
        self._prepare_trainer_defaults()
        # fit these stuffs!
        self.trainer = make_trainer(**shallow_copy_dict(self.trainer_config))
        self.trainer.fit(
            self.loss,
            self.model,
            self.inference,
            self.train_loader,
            self.valid_loader,
            cuda=cuda,
        )
        self.device_info = self.trainer.device_info
        return self

    def predict(
        self,
        x: data_type,
        y: data_type = None,
        *,
        batch_size: int = 128,
        transform_kwargs: Optional[Dict[str, Any]] = None,
        **predict_kwargs: Any,
    ) -> InferenceOutputs:
        loader = MLLoader(
            MLData(*self.data.transform(x, y, **(transform_kwargs or {})).xy),
            shuffle=False,
            batch_size=batch_size,
        )
        return self.inference.get_outputs(self.device, loader, **predict_kwargs)

    def to_pattern(
        self,
        *,
        pre_process: Optional[Callable] = None,
        **predict_kwargs: Any,
    ) -> ModelPattern:
        def _predict(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            outputs = self.predict(x, **predict_kwargs)
            predictions = outputs.forward_results[PREDICTIONS_KEY]
            return np.argmax(predictions, axis=1)[..., None]

        def _predict_prob(x: np.ndarray) -> np.ndarray:
            if pre_process is not None:
                x = pre_process(x)
            outputs = self.predict(x, **predict_kwargs)
            logits = outputs.forward_results[PREDICTIONS_KEY]
            return MetricProtocol.softmax(logits)

        return ModelPattern(
            init_method=lambda: self,
            predict_method=_predict,
            predict_prob_method=_predict_prob,
        )

    def _save_misc(self, export_folder: str, retain_data: bool) -> float:
        # data
        data_folder = os.path.join(export_folder, self.data_folder)
        self.data.save(data_folder, retain_data=retain_data, compress=False)
        # final results
        final_results = self.trainer.final_results
        if final_results is None:
            raise ValueError("`final_results` are not generated yet")
        with open(os.path.join(export_folder, self.final_results_file), "w") as f:
            json.dump(final_results, f)
        return final_results.final_score

    def save(
        self,
        export_folder: str,
        *,
        compress: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
    ) -> "MLPipeline":
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            score = self._save_misc(export_folder, retain_data)
            self.trainer.save_checkpoint(score, export_folder)
            if self.inference is None:
                raise ValueError("`inference` is not yet generated")
            config_bundle = {
                "config": shallow_copy_dict(self.config),
                "device_info": self.device_info,
                "processed_dim": self.processed_dim,
            }
            Saving.save_dict(config_bundle, self.config_bundle_name, export_folder)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)
        return self

    @classmethod
    def load(cls, export_folder: str, *, compress: bool = True) -> "MLPipeline":
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(export_folder, compress):
                config_bundle = Saving.load_dict(cls.config_bundle_name, export_folder)
                config = config_bundle["config"]
                config["in_loading"] = True
                m = cls(**config)
                m.device_info = DeviceInfo(*config_bundle["device_info"])
                m.processed_dim = config_bundle["processed_dim"]
                data_folder = os.path.join(export_folder, cls.data_folder)
                m.data = TabularData.load(data_folder, compress=False)
                m._prepare_modules()
                # restore checkpoint
                score_path = os.path.join(export_folder, SCORES_FILE)
                checkpoints = get_sorted_checkpoints(score_path)
                if not checkpoints:
                    msg = f"{WARNING_PREFIX}no model file found in {export_folder}"
                    raise ValueError(msg)
                checkpoint_path = os.path.join(export_folder, checkpoints[0])
                states = torch.load(checkpoint_path, map_location=m.device)
                if m.encoder is not None:
                    encoder_cache_keys = []
                    for key in states:
                        if key.startswith("encoder") and key.endswith("cache"):
                            encoder_cache_keys.append(key)
                    for key in encoder_cache_keys:
                        states.pop(key)
                m.model.load_state_dict(states)
        return m


__all__ = [
    "MLPipeline",
]
