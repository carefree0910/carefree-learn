from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional
from cfdata.tabular.api import TabularData

from ...types import data_type
from ...types import sample_weights_type
from ..internal_.pipeline import _split_sw
from ...misc.internal_ import MLLoader
from ...misc.internal_ import MLDataset
from ...misc.internal_ import DLDataModule


class MLData(DLDataModule):
    def __init__(
        self,
        x_train: data_type,
        y_train: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        cf_data: Optional[TabularData] = None,
        num_history: int = 1,
        is_classification: Optional[bool] = None,
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
        # inference
        for_inference: bool = False,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.cf_data = cf_data
        self.num_history = num_history
        if is_classification is None and cf_data is None and not for_inference:
            msg = "`cf_data` should be provided when `is_classification` is None"
            raise ValueError(msg)
        self.is_classification = is_classification
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
    def json(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "num_history": self.num_history,
            "is_classification": self.is_classification,
        }

    def prepare(self, sample_weights: sample_weights_type) -> None:
        self.train_weights, self.valid_weights = _split_sw(sample_weights)
        if self.cf_data is not None:
            self.cf_data.read(self.x_train, self.y_train, **self.read_config)
            if self.is_classification is None:
                self.is_classification = self.cf_data.is_clf
            if self.x_valid is not None:
                self.train_cf_data = self.cf_data
                self.valid_cf_data = self.cf_data.copy_to(self.x_valid, self.y_valid)
            else:
                if isinstance(self.valid_split, int):
                    split = self.valid_split
                else:
                    num_data = len(self.cf_data)
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
                    self.train_cf_data = self.cf_data
                    self.valid_cf_data = None
                else:
                    rs = self.cf_data.split(split, order=self.valid_split_order)
                    self.train_cf_data = rs.remained
                    self.valid_cf_data = rs.split
            self.train_data = MLDataset(*self.train_cf_data.processed.xy)
            self.valid_data = MLDataset(*self.valid_cf_data.processed.xy)
            self.num_classes = self.train_cf_data.num_classes
            self.input_dim = self.train_cf_data.processed_dim
            return None
        if isinstance(self.x_train, str):
            raise ValueError("`cf_data` should be provided when `x_train` is `str`")
        self.num_classes = None
        self.input_dim = self.x_train.shape[-1]
        self.train_data = MLDataset(self.x_train, self.y_train)
        if self.x_valid is None or self.y_valid is None:
            self.valid_data = None
        else:
            self.valid_data = MLDataset(self.x_valid, self.y_valid)

    def initialize(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        train_loader = MLLoader(
            self.train_data,
            name="train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            sample_weights=self.train_weights,
        )
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = MLLoader(
                self.valid_data,
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
                sample_weights=self.valid_weights,
            )
        return train_loader, valid_loader

    @classmethod
    def with_cf_data(
        cls,
        *args: Any,
        is_classification: Optional[bool] = None,
        data_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLData":
        if data_config is None:
            data_config = {}
        data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            data_config["task_type"] = "clf" if is_classification else "reg"
        kwargs["is_classification"] = is_classification
        kwargs["cf_data"] = TabularData(**(data_config or {}))
        return cls(*args, **kwargs)


class MLInferenceData(MLData):
    def __init__(self, x: data_type, y: data_type = None):
        super().__init__(x, y, for_inference=True)


__all__ = [
    "MLData",
    "MLInferenceData",
]
