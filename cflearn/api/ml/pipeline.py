import torch

from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from cfdata.tabular import TabularData

from ...types import data_type
from ...trainer import Trainer
from ...trainer import DeviceInfo
from ...protocol import loss_dict
from ...protocol import model_dict
from ...protocol import LossProtocol
from ...protocol import InferenceOutputs
from ..internal_.trainer import make_trainer
from ...misc.internal_ import MLData
from ...misc.internal_ import MLLoader
from ...misc.internal_ import MLInference
from ...models.ml.protocol import MLModelProtocol


class MLPipeline:
    data: TabularData
    loss: LossProtocol
    model: MLModelProtocol
    trainer: Trainer
    inference: MLInference
    device_info: DeviceInfo

    def __init__(
        self,
        model_name: str = "fcnn",
        model_config: Optional[Dict[str, Any]] = None,
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
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
    ):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.loss_name = loss_name
        self.loss_config = loss_config or {}
        if data_config is None:
            data_config = {}
        data_config["default_categorical_process"] = "identical"
        data_config.setdefault("use_timing_context", True)
        self.data = TabularData(**(data_config or {}))
        self.read_config = read_config or {}
        self.valid_split = valid_split
        self.min_valid_split = min_valid_split
        self.max_valid_split = max_valid_split
        self.max_valid_split_ratio = max_valid_split_ratio
        self.valid_split_order = valid_split_order
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

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
        metric_log_file: str = "metrics.txt",
        rank: Optional[int] = None,
    ) -> "MLPipeline":
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
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
        )
        if valid_data is None:
            valid_loader = None
        else:
            valid_loader = MLLoader(
                MLData(*valid_data.processed.xy),
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
            )
        # prepare
        if self.loss_name == "auto":
            self.loss_name = "focal" if self.data.is_clf else "mae"
        loss = loss_dict[self.loss_name](**(self.loss_config or {}))
        self.model_config["in_dim"] = self.data.processed_dim
        self.model_config["out_dim"] = 1 if self.data.is_reg else self.data.num_classes
        model = model_dict[self.model_name](**self.model_config)
        if not isinstance(model, MLModelProtocol):
            raise ValueError(f"'{self.model_name}' is not an ML model")
        self.model = model
        inference = self.inference = MLInference(model)
        # set some defaults to ml tasks which work well in practice
        if metric_names is None and self.data.is_clf:
            metric_names = ["acc", "auc"]
        if monitor_names is None:
            monitor_names = ["mean_std", "plateau"]
        if optimizer_settings is None:
            optimizer_settings = {"all": {"optimizer": "adam", "scheduler": "warmup"}}
        # fit these stuffs!
        self.trainer = make_trainer(
            state_config,
            num_epoch=num_epoch,
            valid_portion=valid_portion,
            amp=amp,
            clip_norm=clip_norm,
            metric_names=metric_names,
            metric_configs=metric_configs,
            monitor_names=monitor_names,
            monitor_configs=monitor_configs,
            callback_names=callback_names,
            callback_configs=callback_configs,
            optimizer_settings=optimizer_settings,
            workplace=workplace,
            metric_log_file=metric_log_file,
            rank=rank,
        )
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


__all__ = [
    "MLPipeline",
]
