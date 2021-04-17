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
from cftool.misc import timestamp
from cftool.misc import shallow_copy_dict

from ...types import data_type
from ...trainer import Trainer
from ...trainer import DeviceInfo
from ...protocol import loss_dict
from ...protocol import LossProtocol
from ...protocol import MetricProtocol
from ...protocol import InferenceOutputs
from ...constants import PREDICTIONS_KEY
from ..internal_.trainer import make_trainer
from ...misc.toolkit import get_arguments
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

    metrics_log_file: str = "metrics.txt"

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
        configs_file: str = "configs.json",
        rank: Optional[int] = None,
        tqdm_settings: Optional[Dict[str, Any]] = None,
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
        self.configs_file = configs_file
        self.trainer_config = {
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

    @property
    def device(self) -> torch.device:
        return self.device_info.device

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
        workplace = self.trainer_config["workplace"] = os.path.join(
            self.trainer_config["workplace"],
            timestamp(ensure_different=True),
        )
        self.trainer_config["metrics_log_file"] = self.metrics_log_file
        os.makedirs(workplace, exist_ok=True)
        with open(os.path.join(workplace, self.configs_file), "w") as f:
            json.dump(self.config, f)
        self.data.read(x, y, **self.read_config)
        # split data
        if x_valid is not None:
            train_data = self.data
            valid_data = self.data.copy_to(x_valid, y_valid)
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
                train_data = MLData(*self.data.processed.xy)
                valid_data = None
            else:
                split_result = self.data.split(split, order=self.valid_split_order)
                train_data = split_result.remained
                valid_data = split_result.split
        train_loader = MLLoader(
            MLData(*train_data.processed.xy),
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
        )
        train_loader_copy = copy.deepcopy(train_loader)
        train_loader_copy.shuffle = False
        if valid_data is None:
            valid_loader = train_loader_copy
        else:
            valid_loader = MLLoader(
                MLData(*valid_data.processed.xy),
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
            )
        # set properties
        self.train_data = train_data
        self.valid_data = valid_data
        self.train_loader = train_loader
        self.train_loader_copy = train_loader_copy
        self.valid_loader = valid_loader
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
            loaders = [train_loader_copy]
            if valid_loader is not None:
                loaders.append(valid_loader)
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
        loss = loss_dict[self.loss_name](**(self.loss_config or {}))
        model = self.model = MLModel(
            self.data.processed_dim,
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
        inference = self.inference = MLInference(model)
        # set some defaults to ml tasks which work well in practice
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
        # fit these stuffs!
        self.trainer = make_trainer(**shallow_copy_dict(self.trainer_config))
        self.trainer.fit(loss, model, inference, train_loader, valid_loader, cuda=cuda)
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


__all__ = [
    "MLPipeline",
]
