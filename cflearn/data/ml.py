import os
import json
import tempfile

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cftool.misc import Saving
from cftool.misc import WithRegister
from cftool.array import to_torch
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .core import DLDataModule
from ..types import data_type
from ..types import sample_weights_type
from ..protocol import IDataset
from ..protocol import IDataLoader
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import ConfigMeta

try:
    from cfdata.tabular.api import TabularData
except:
    TabularData = None


ml_loader_callbacks: Dict[str, Type["IMLLoaderCallback"]] = {}
ml_data_modifiers: Dict[str, Type["IMLDataModifier"]] = {}


def get_weighted_indices(
    n: int,
    weights: Optional[np.ndarray],
    ensure_all_occur: bool = False,
) -> np.ndarray:
    indices = np.arange(n)
    if weights is not None:
        numbers = np.random.multinomial(n, weights)
        if ensure_all_occur:
            numbers += 1
        indices = indices.repeat(numbers)
    return indices


class IMLDataset(IDataset, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        pass


class IMLLoaderCallback(WithRegister["IMLLoaderCallback"], metaclass=ABCMeta):
    d = ml_loader_callbacks

    def __init__(self, loader: "IMLLoader"):
        self.loader = loader

    @property
    def data(self) -> IMLDataset:
        return self.loader.data

    # changes may happen inplace, but the final batch will be returned anyway
    @abstractmethod
    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        pass


class IMLLoader(IDataLoader, metaclass=ABCMeta):
    callback: str

    cursor: int
    indices: np.ndarray

    data: IMLDataset
    shuffle: bool
    shuffle_backup: bool
    name: Optional[str]
    batch_size: int
    use_numpy: bool

    def __init__(
        self,
        data: IMLDataset,
        shuffle: bool,
        *,
        name: Optional[str] = None,
        batch_size: int = 128,
        sample_weights: Optional[np.ndarray] = None,
        use_numpy: bool = False,
    ):
        if sample_weights is not None and len(data) != len(sample_weights):
            raise ValueError(
                f"the number of data samples ({len(data)}) is not identical with "
                f"the number of sample weights ({len(sample_weights)})"
            )
        super().__init__(sample_weights=sample_weights)
        self.data = data
        self.shuffle = shuffle
        self.shuffle_backup = shuffle
        self.name = name
        self.batch_size = batch_size
        self.use_numpy = use_numpy

    def __iter__(self) -> "IMLLoader":
        self.cursor = 0
        self.indices = get_weighted_indices(len(self.data), self.sample_weights)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Union[np_dict_type, tensor_dict_type]:
        start = self.cursor
        if start >= len(self.data):
            raise StopIteration
        self.cursor += self.batch_size
        indices = self.indices[start : self.cursor]
        batch = self.data[indices]
        batch = self._make_callback().process_batch(batch)
        batch.setdefault(BATCH_INDICES_KEY, indices)
        if self.use_numpy:
            return batch
        return {k: None if v is None else to_torch(v) for k, v in batch.items()}

    def _make_callback(self) -> IMLLoaderCallback:
        return IMLLoaderCallback.make(self.callback, {"loader": self})

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "IMLLoader":
        return self.__class__(
            self.data,
            self.shuffle,
            name=self.name,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights,
        )


class IMLDataInfo(NamedTuple):
    """
    * input_dim (int) : final dim that the model will receive
    * num_history (int) : number of history, useful in time series tasks
    * num_classes (int | None) : number of classes
      -> will be used as `output_dim` if `is_classification` is True & `output_dim` is not specified
    * is_classification (bool | None) : whether current task is a classification task
      -> it should always be provided unless it's at inference time
    """

    input_dim: int
    num_history: int
    num_classes: Optional[int] = None
    is_classification: Optional[bool] = None


class IMLDataModifier(WithRegister["IMLDataModifier"], metaclass=ABCMeta):
    d = ml_data_modifiers

    def __init__(self, data: "IMLData"):
        self.data = data

    # abstract

    @abstractmethod
    def save(self, data_folder: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, data_folder: str, info: Dict[str, Any]) -> "IMLData":
        pass

    # optional callbacks

    # changes should happen inplace
    def permute_info_in_save(self, info: Dict[str, Any]) -> None:
        pass

    # changes should happen inplace
    @classmethod
    def permute_info_in_load(
        cls,
        data_cls: Type["IMLData"],
        info: Dict[str, Any],
    ) -> None:
        pass


class IMLData(DLDataModule, metaclass=ABCMeta):
    modifier: str

    config: Dict[str, Any]

    @abstractmethod
    def get_info(self) -> IMLDataInfo:
        pass

    @property
    def info(self) -> Dict[str, Any]:
        return self.get_info()._asdict()

    def _make_modifier(self) -> IMLDataModifier:
        return IMLDataModifier.make(self.modifier, {"data": self})

    def _save_info(self, folder: str) -> None:
        info = self.info
        self._make_modifier().permute_info_in_save(info)
        Saving.save_dict(info, self.info_name, folder)

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        info = super()._load_info(folder)
        IMLDataModifier.get(cls.modifier).permute_info_in_load(cls, info)
        return info

    def _save_data(self, data_folder: str) -> None:
        self._make_modifier().save(data_folder)

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "IMLData":
        data = IMLDataModifier.get(cls.modifier).load(data_folder, info)
        data.loaded = True
        data.prepare(sample_weights)
        return data


def register_ml_loader_callback(
    name: str,
    *,
    allow_duplicate: bool = False,
) -> Callable:
    return IMLLoaderCallback.register(name, allow_duplicate=allow_duplicate)


def register_ml_data_modifier(name: str, *, allow_duplicate: bool = False) -> Callable:
    return IMLDataModifier.register(name, allow_duplicate=allow_duplicate)


@IDataset.register("ml")
class MLDataset(IMLDataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray], **others: np.ndarray):
        super().__init__()
        self.x = x
        self.y = y
        self.others = others

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        batch = {
            INPUT_KEY: self.x[item],
            LABEL_KEY: None if self.y is None else self.y[item],
        }
        for k, v in self.others.items():
            batch[k] = v[item]
        return batch

    def __len__(self) -> int:
        return len(self.x)


@IMLLoaderCallback.register("basic")
class BasicMLLoaderCallback(IMLLoaderCallback):
    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        return batch


@IDataLoader.register("ml")
class MLLoader(IMLLoader):
    callback = "basic"


# api


split_sw_type = Tuple[Optional[np.ndarray], Optional[np.ndarray]]


def _norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def _split_sw(sample_weights: sample_weights_type) -> split_sw_type:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(_norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


@IMLDataModifier.register("_internal.ml")
class _InternalMLDataModifier(IMLDataModifier):
    data: "MLData"

    def save(self, data_folder: str) -> None:
        with open(os.path.join(data_folder, self.data.arguments_file), "w") as f:
            json.dump(self.data.config, f)
        all_data = [
            self.data.x_train,
            self.data.y_train,
            self.data.x_valid,
            self.data.y_valid,
        ]
        for data, file in zip(all_data, self.data.data_files):
            if data is not None:
                np.save(os.path.join(data_folder, file), data)
        if self.data.train_others is not None:
            for k, v in self.data.train_others.items():
                np.save(os.path.join(data_folder, f"{k}_train.npy"), v)
        if self.data.valid_others is not None:
            for k, v in self.data.valid_others.items():
                np.save(os.path.join(data_folder, f"{k}_valid.npy"), v)
        if self.data.cf_data is not None:
            full_cf_data_folder = os.path.join(data_folder, self.data.full_cf_data_name)
            self.data.cf_data.save(full_cf_data_folder, retain_data=True)

    @classmethod
    def load(cls, data_folder: str, info: Dict[str, Any]) -> "MLData":
        args = []
        for file in MLData.data_files:
            path = os.path.join(data_folder, file)
            args.append(None if not os.path.isfile(path) else np.load(path))
        with open(os.path.join(data_folder, MLData.arguments_file), "r") as f:
            kwargs = json.load(f)
        train_others = {}
        valid_others = {}
        for file in os.listdir(data_folder):
            if file in MLData.data_files:
                continue
            path = os.path.join(data_folder, file)
            if file.endswith("_train"):
                train_others[file.split("_train")[0]] = np.load(path)
            elif file.endswith("_valid"):
                valid_others[file.split("_valid")[0]] = np.load(path)
        if train_others:
            kwargs["train_others"] = train_others
        if valid_others:
            kwargs["valid_others"] = valid_others
        if info["cf_data"] is not None:
            if TabularData is None:
                raise ValueError(
                    "`carefree-data` needs to be installed "
                    "to load `MLData` with `cf_data` defined"
                )
            full_cf_data_folder = os.path.join(data_folder, MLData.full_cf_data_name)
            kwargs["cf_data"] = TabularData.load(full_cf_data_folder)
        return MLData(*args, **kwargs)

    def permute_info_in_save(self, info: Dict[str, Any]) -> None:
        if info["cf_data"] is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_name = os.path.join(tmp_dir, self.data.tmp_cf_data_name)
                info["cf_data"].save(tmp_name, retain_data=False)
                zip_file = f"{tmp_name}.zip"
                with open(zip_file, "rb") as f:
                    info["cf_data"] = f.read()
                os.remove(zip_file)

    @classmethod
    def permute_info_in_load(
        cls,
        data_cls: Type["MLData"],
        info: Dict[str, Any],
    ) -> None:
        cf_data = info["cf_data"]
        if cf_data is None:
            return
        if TabularData is None:
            raise ValueError(
                "`carefree-data` needs to be installed "
                "to load `MLData` with `cf_data` defined"
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = os.path.join(tmp_dir, data_cls.tmp_cf_data_name)
            zip_file = f"{tmp_name}.zip"
            with open(zip_file, "wb") as f:
                f.write(cf_data)
            info["cf_data"] = TabularData.load(tmp_name)
            os.remove(zip_file)


@DLDataModule.register("ml")
class MLData(IMLData, metaclass=ConfigMeta):
    modifier = "_internal.ml"

    cf_data: Optional[TabularData]
    train_data: MLDataset
    valid_data: Optional[MLDataset]
    train_cf_data: Optional[TabularData]
    valid_cf_data: Optional[TabularData]

    data_files = [
        "x_train.npy",
        "y_train.npy",
        "x_valid.npy",
        "y_valid.npy",
    ]
    arguments_file = "arguments.json"
    tmp_cf_data_name = ".tmp_cf_data"
    full_cf_data_name = "cf_data"

    def __init__(
        self,
        x_train: data_type,
        y_train: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
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
        contains_labels: bool = True,
    ):
        pop_keys = [
            "x_train",
            "y_train",
            "x_valid",
            "y_valid",
            "train_others",
            "valid_others",
            "cf_data",
        ]
        for key in pop_keys:
            self.config.pop(key, None)
        assert x_train is not None
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.train_others = train_others
        self.valid_others = valid_others
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
        self.for_inference = for_inference
        self.contains_labels = contains_labels
        self.loaded = False

    def get_info(self) -> IMLDataInfo:
        return IMLDataInfo(
            self.input_dim,
            self.num_history,
            self.num_classes,
            self.is_classification,
        )

    @property
    def info(self) -> Dict[str, Any]:
        info = super().info
        info["cf_data"] = self.cf_data
        return info

    def prepare(self, sample_weights: sample_weights_type) -> None:
        train_others = self.train_others or {}
        valid_others = self.valid_others or {}
        self.train_weights, self.valid_weights = _split_sw(sample_weights)
        if self.cf_data is not None:
            if self.for_inference:
                train_xy = self.cf_data.transform(
                    self.x_train,
                    None,
                    contains_labels=self.contains_labels,
                ).xy
                self.train_data = MLDataset(*train_xy)
                self.valid_data = None
                self.train_cf_data = self.cf_data
                self.valid_cf_data = None
                # if `for_inference` is True, these properties are not needed
                self.input_dim = -1
                self.num_classes = self.is_classification = None
            else:
                if not self.loaded:
                    self.cf_data.read(self.x_train, self.y_train, **self.read_config)
                if self.x_valid is not None:
                    self.train_cf_data = self.cf_data
                    self.valid_cf_data = self.cf_data.copy_to(
                        self.x_valid, self.y_valid
                    )
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
                            max_split = int(
                                round(num_data * self.max_valid_split_ratio)
                            )
                            max_split = min(max_split, self.max_valid_split)
                            split = min(num_split, max_split)
                    if split <= 0:
                        self.train_cf_data = self.cf_data
                        self.valid_cf_data = None
                    else:
                        rs = self.cf_data.split(split, order=self.valid_split_order)
                        self.train_cf_data = rs.remained
                        self.valid_cf_data = rs.split
                train_xy = self.train_cf_data.processed.xy
                self.train_data = MLDataset(*train_xy, **train_others)
                if self.valid_cf_data is None:
                    self.valid_data = None
                else:
                    valid_xy = self.valid_cf_data.processed.xy
                    self.valid_data = MLDataset(*valid_xy, **valid_others)
                # initialize properties with train_cf_data
                self.input_dim = self.train_cf_data.processed_dim
                self.num_classes = self.train_cf_data.num_classes
                if self.is_classification is None:
                    self.is_classification = self.train_cf_data.is_clf
            return None
        if isinstance(self.x_train, str):
            raise ValueError("`cf_data` should be provided when `x_train` is `str`")
        self.num_classes = None
        self.input_dim = self.x_train.shape[-1]
        self.train_data = MLDataset(self.x_train, self.y_train, **train_others)
        if self.x_valid is None or self.y_valid is None:
            self.valid_data = None
        else:
            self.valid_data = MLDataset(self.x_valid, self.y_valid, **valid_others)

    def initialize(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        train_loader = MLLoader(
            self.train_data,
            name=None if self.for_inference else "train",
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
        cf_data_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLData":
        if TabularData is None:
            raise ValueError("`carefree-data` needs to be installed for `with_cf_data`")
        if cf_data_config is None:
            cf_data_config = {}
        cf_data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            cf_data_config["task_type"] = "clf" if is_classification else "reg"
        kwargs["is_classification"] = is_classification
        kwargs["cf_data"] = TabularData(**(cf_data_config or {}))
        return cls(*args, **kwargs)


class MLInferenceData(MLData):
    def __init__(
        self,
        x: data_type,
        y: data_type = None,
        *,
        shuffle: bool = False,
        contains_labels: bool = True,
        cf_data: Optional[TabularData] = None,
    ):
        super().__init__(
            x,
            y,
            cf_data=cf_data,
            shuffle_train=shuffle,
            for_inference=True,
            contains_labels=contains_labels,
        )
        self.prepare(None)


__all__ = [
    "get_weighted_indices",
    "register_ml_loader_callback",
    "register_ml_data_modifier",
    "IMLDataset",
    "IMLLoaderCallback",
    "IMLLoader",
    "IMLDataInfo",
    "IMLDataModifier",
    "IMLData",
    "MLDataset",
    "MLLoader",
    "MLData",
    "MLInferenceData",
]
