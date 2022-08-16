import os
import json
import shutil
import hashlib
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
from cftool.misc import hash_code
from cftool.misc import Saving
from cftool.misc import WithRegister
from cftool.array import to_torch
from cftool.array import is_string
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .core import DLDataModule
from ..types import data_type
from ..types import sample_weights_type
from ..protocol import IDataset
from ..protocol import IDataLoader
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import DATA_CACHE_DIR
from ..constants import BATCH_INDICES_KEY
from ..misc.toolkit import ConfigMeta

try:
    from cfdata.tabular.api import TabularData
except:
    TabularData = None


ml_data_processors: Dict[str, Type["IMLDataProcessor"]] = {}


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


# protocols


class IMLDataInfo(NamedTuple):
    """
    * input_dim (int) : input feature dim that the model will receive
    * num_history (Optional[int]) : number of history, useful in time series tasks
    * num_classes (Optional[int]) : number of classes
      -> will be used as `output_dim` if `is_classification` is True & `output_dim` is not specified
    * is_classification (Optional[bool]) : whether current task is a classification task
    """

    input_dim: int
    num_history: Optional[int] = None
    num_classes: Optional[int] = None
    is_classification: Optional[bool] = None


class IMLPreProcessedData(NamedTuple):
    x_train: np.ndarray
    y_train: Optional[np.ndarray]
    x_valid: Optional[np.ndarray]
    y_valid: Optional[np.ndarray]
    data_info: IMLDataInfo


class IMLDataProcessor(WithRegister["IMLDataProcessor"], metaclass=ABCMeta):
    d = ml_data_processors

    is_ready: bool = False

    cache_folder = os.path.join(DATA_CACHE_DIR, "IMLDataProcessor")

    # abstract

    @abstractmethod
    def build_with(
        self,
        data: "IMLData",
        x_train: Union[np.ndarray, str],
        y_train: Optional[Union[np.ndarray, str]],
        x_valid: Optional[Union[np.ndarray, str]],
        y_valid: Optional[Union[np.ndarray, str]],
    ) -> None:
        pass

    @abstractmethod
    def preprocess(
        self,
        data: "IMLData",
        x_train: Union[np.ndarray, str],
        y_train: Optional[Union[np.ndarray, str]],
        x_valid: Optional[Union[np.ndarray, str]],
        y_valid: Optional[Union[np.ndarray, str]],
        *,
        for_inference: bool,
    ) -> IMLPreProcessedData:
        pass

    @abstractmethod
    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        pass

    @abstractmethod
    def dumps(self) -> Any:
        pass

    @abstractmethod
    def loads(self, dumped: Any) -> None:
        pass

    # api

    ## caching

    @staticmethod
    def get_hash(data: Optional[Union[np.ndarray, Any]]) -> Optional[str]:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return hashlib.sha1(data).hexdigest()
        return hash_code(str(data))

    @classmethod
    def clear_all_cache(cls) -> None:
        shutil.rmtree(cls.cache_folder)

    ## serialization

    def to_pack(self) -> Dict[str, Any]:
        return dict(type=self.__identifier__, info=self.dumps())

    @classmethod
    def from_pack(cls, pack: Dict[str, Any]) -> "IMLDataProcessor":
        processor = cls.get(pack["type"])()
        processor.loads(pack["info"])
        processor.is_ready = True
        return processor


def register_ml_data_processor(name: str, *, allow_duplicate: bool = False) -> Callable:
    return IMLDataProcessor.register(name, allow_duplicate=allow_duplicate)


class IMLDataset(IDataset, metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        pass


class IMLLoader(IDataLoader, metaclass=ABCMeta):
    callback: str

    cursor: int
    indices: np.ndarray

    data: IMLDataset
    processor: IMLDataProcessor
    shuffle: bool
    shuffle_backup: bool
    name: Optional[str]
    batch_size: int
    use_numpy: bool

    def __init__(
        self,
        data: IMLDataset,
        processor: IMLDataProcessor,
        *,
        shuffle: bool,
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
        self.processor = processor
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
        batch = self.processor.process_batch(batch)
        batch.setdefault(BATCH_INDICES_KEY, indices)
        if self.use_numpy:
            return batch
        return {k: None if v is None else to_torch(v) for k, v in batch.items()}

    def disable_shuffle(self) -> None:
        self.shuffle = False

    def recover_shuffle(self) -> None:
        self.shuffle = self.shuffle_backup

    def copy(self) -> "IMLLoader":
        return self.__class__(
            self.data,
            self.processor,
            name=self.name,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            sample_weights=self.sample_weights,
        )


# internal


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


@IDataLoader.register("ml")
class MLLoader(IMLLoader):
    pass


class IMLPreProcessedXY(NamedTuple):
    x: np.ndarray
    y: Optional[np.ndarray]


class IMLData(DLDataModule, metaclass=ConfigMeta):
    config: Dict[str, Any]

    processor_type: str
    processor_info: Optional[Dict[str, Any]] = None

    train_data: MLDataset
    valid_data: Optional[MLDataset]

    data_files = [
        "x_train.npy",
        "y_train.npy",
        "x_valid.npy",
        "y_valid.npy",
    ]
    arguments_file = "arguments.json"

    input_dim: int
    num_history: int
    num_classes: Optional[int]
    is_classification: Optional[bool]

    processor: Optional[IMLDataProcessor]
    train_others: Optional[np_dict_type]
    valid_others: Optional[np_dict_type]

    shuffle_train: bool
    shuffle_valid: bool
    batch_size: int
    valid_batch_size: int
    train_weights: Optional[np.ndarray]
    valid_weights: Optional[np.ndarray]

    for_inference: bool

    def __init__(
        self,
        x_train: Any,
        y_train: Optional[Any] = None,
        x_valid: Optional[Any] = None,
        y_valid: Optional[Any] = None,
        *,
        # processor
        processor: Optional[IMLDataProcessor] = None,
        # auxiliary data
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        # common
        num_history: int = 1,
        num_classes: Optional[int] = None,
        is_classification: Optional[bool] = None,
        # data loader
        shuffle_train: bool = True,
        shuffle_valid: bool = False,
        batch_size: int = 128,
        valid_batch_size: int = 512,
        # inference
        for_inference: bool = False,
    ):
        if for_inference:
            msg = None
            fmt = "`processor` should be {} when `for_inference` is True"
            if processor is None:
                msg = fmt.format("provided")
            elif not processor.is_ready:
                msg = fmt.format("ready")
            if msg is not None:
                raise ValueError(msg)
        pop_keys = [
            "x_train",
            "y_train",
            "x_valid",
            "y_valid",
            "processor",
            "train_others",
            "valid_others",
        ]
        for key in pop_keys:
            self.config.pop(key, None)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.processor = processor
        self.train_others = train_others
        self.valid_others = valid_others
        self.num_history = num_history
        self.num_classes = num_classes
        self.is_classification = is_classification
        self.shuffle_train = shuffle_train
        self.shuffle_valid = shuffle_valid
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.for_inference = for_inference

    # api

    def preprocess(
        self,
        x: Union[np.ndarray, str],
        y: Optional[Union[np.ndarray, str]],
    ) -> IMLPreProcessedXY:
        if self.processor is None or not self.processor.is_ready:
            raise ValueError("`processor` should be ready before calling `preprocess`")
        res = self.processor.preprocess(self, x, y, None, None, for_inference=True)
        return IMLPreProcessedXY(res.x_train, res.y_train)

    # inheritance

    @property
    def info(self) -> Dict[str, Any]:
        if self.processor is None:
            raise ValueError(
                "`processor` should be provided before accessing `info`, "
                "did you forget to call the `prepare` method first?"
            )
        return {
            "type": self.__identifier__,
            "processor": self.processor,
            "input_dim": self.input_dim,
            "num_history": self.num_history,
            "num_classes": self.num_classes,
            "is_classification": self.is_classification,
        }

    def prepare(self, sample_weights: sample_weights_type) -> None:
        train_others = self.train_others or {}
        valid_others = self.valid_others or {}
        self.train_weights, self.valid_weights = _split_sw(sample_weights)
        data_args = self, self.x_train, self.y_train, self.x_valid, self.y_valid
        if self.for_inference:
            assert self.processor is not None
            processor = self.processor
        elif self.processor is not None and self.processor.is_ready:
            processor = self.processor
        else:
            processor = IMLDataProcessor.get(self.processor_type)()
            processor.build_with(*data_args)
            self.processor = processor
        final = processor.preprocess(*data_args, for_inference=self.for_inference)
        for k, v in final.data_info._asdict().items():
            if v is not None:
                setattr(self, k, v)
        self.train_data = MLDataset(final.x_train, final.y_train, **train_others)
        if final.x_valid is None or final.y_valid is None:
            self.valid_data = None
        else:
            self.valid_data = MLDataset(final.x_valid, final.y_valid, **valid_others)

    def initialize(self) -> Tuple[MLLoader, Optional[MLLoader]]:
        if self.processor is None:
            raise ValueError(
                "`processor` should be provided before calling `initialize`, "
                "did you forget to call the `prepare` method first?"
            )
        train_loader = MLLoader(
            self.train_data,
            self.processor,
            name=None if self.for_inference else "train",
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            sample_weights=self.train_weights,
        )
        if self.valid_data is None:
            valid_loader = None
        else:
            # when `for_inference` is True, `valid_data` will always be `None`
            # so we don't need to condition `name` field here
            valid_loader = MLLoader(
                self.valid_data,
                self.processor,
                name="valid",
                shuffle=self.shuffle_valid,
                batch_size=self.valid_batch_size,
                sample_weights=self.valid_weights,
            )
        return train_loader, valid_loader

    def _save_info(self, folder: str) -> None:
        info = self.info
        info["processor"] = info["processor"].to_pack()
        Saving.save_dict(info, self.info_name, folder)

    def _save_data(self, data_folder: str) -> None:
        with open(os.path.join(data_folder, self.arguments_file), "w") as f:
            json.dump(self.config, f)
        all_data = [
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
        ]
        for data, file in zip(all_data, self.data_files):
            if data is not None:
                np.save(os.path.join(data_folder, file), data)
        if self.train_others is not None:
            for k, v in self.train_others.items():
                np.save(os.path.join(data_folder, f"{k}_train.npy"), v)
        if self.valid_others is not None:
            for k, v in self.valid_others.items():
                np.save(os.path.join(data_folder, f"{k}_valid.npy"), v)

    @classmethod
    def _get_load_arguments(cls, data_folder: str) -> Tuple[List[Any], Dict[str, Any]]:
        args = []
        for file in cls.data_files:
            path = os.path.join(data_folder, file)
            if not os.path.isfile(path):
                data = None
            else:
                data = np.load(path)
                if is_string(data):
                    data = data.item()
            args.append(data)
        with open(os.path.join(data_folder, cls.arguments_file), "r") as f:
            kwargs = json.load(f)
        train_others = {}
        valid_others = {}
        for file in os.listdir(data_folder):
            if file in cls.data_files:
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
        return args, kwargs

    @classmethod
    def _load_info(cls, folder: str) -> Dict[str, Any]:
        info = super()._load_info(folder)
        processor_pack = info.pop("processor")
        processor = IMLDataProcessor.from_pack(processor_pack)
        info["processor"] = processor
        return info

    @classmethod
    def _load(
        cls,
        data_folder: str,
        info: Dict[str, Any],
        sample_weights: sample_weights_type,
    ) -> "IMLData":
        args, kwargs = cls._get_load_arguments(data_folder)
        kwargs["processor"] = info["processor"]
        data = cls(*args, **kwargs)
        data.prepare(sample_weights)
        return data


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


@IMLDataProcessor.register("_internal.basic")
class _InternalBasicMLDataProcessor(IMLDataProcessor):
    def build_with(
        self,
        data: "IMLData",
        x_train: np.ndarray,
        y_train: Optional[np.ndarray],
        x_valid: Optional[np.ndarray],
        y_valid: Optional[np.ndarray],
    ) -> None:
        pass

    def preprocess(
        self,
        data: "MLData",
        x_train: np.ndarray,
        y_train: Optional[np.ndarray],
        x_valid: Optional[np.ndarray],
        y_valid: Optional[np.ndarray],
        *,
        for_inference: bool,
    ) -> IMLPreProcessedData:
        return IMLPreProcessedData(
            x_train,
            y_train,
            x_valid,
            y_valid,
            IMLDataInfo(x_train.shape[-1]),
        )

    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        return batch

    def dumps(self) -> Any:
        return {}

    def loads(self, dumped: Any) -> None:
        pass


@DLDataModule.register("ml")
class MLData(IMLData):
    processor_type = "_internal.basic"


class MLInferenceData(MLData):
    def __init__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        shuffle: bool = False,
    ):
        processor = _InternalBasicMLDataProcessor()
        processor.is_ready = True
        super().__init__(
            x,
            y,
            processor=processor,
            shuffle_train=shuffle,
            for_inference=True,
        )
        self.prepare(None)


@IMLDataProcessor.register("_internal.carefree")
class _InternalCarefreeMLDataProcessor(IMLDataProcessor):
    cf_data: TabularData

    tmp_cf_data_name = ".tmp_cf_data"
    full_cf_data_name = "cf_data"

    def build_with(
        self,
        data: "MLCarefreeData",
        x_train: Union[np.ndarray, str],
        y_train: Optional[Union[np.ndarray, str]],
        x_valid: Optional[Union[np.ndarray, str]],
        y_valid: Optional[Union[np.ndarray, str]],
    ) -> None:
        self.cf_data = TabularData(**data.data_config)
        self.cf_data.read(x_train, y_train, **data.read_config)

    def preprocess(
        self,
        data: "MLCarefreeData",
        x_train: Union[np.ndarray, str],
        y_train: Optional[Union[np.ndarray, str]],
        x_valid: Optional[Union[np.ndarray, str]],
        y_valid: Optional[Union[np.ndarray, str]],
        *,
        for_inference: bool,
    ) -> IMLPreProcessedData:
        if for_inference:
            x_train, y_train = self.cf_data.transform(
                x_train,
                y_train,
                contains_labels=data.contains_labels,
            ).xy
            return IMLPreProcessedData(x_train, y_train, None, None, IMLDataInfo(-1))
        # split data
        if x_valid is not None:
            train_cf_data = self.cf_data
            valid_cf_data = self.cf_data.copy_to(x_valid, y_valid)
        else:
            if isinstance(data.valid_split, int):
                split = data.valid_split
            else:
                num_data = len(self.cf_data)
                if isinstance(data.valid_split, float):
                    split = int(round(data.valid_split * num_data))
                else:
                    default_split = 0.1
                    num_split = int(round(default_split * num_data))
                    num_split = max(data.min_valid_split, num_split)
                    max_split = int(round(num_data * data.max_valid_split_ratio))
                    max_split = min(max_split, data.max_valid_split)
                    split = min(num_split, max_split)
            if split <= 0:
                train_cf_data = self.cf_data
                valid_cf_data = None
            else:
                rs = self.cf_data.split(split, order=data.valid_split_order)
                train_cf_data = rs.remained
                valid_cf_data = rs.split
        # process data
        if train_cf_data.processed is not None:
            x_train, y_train = train_cf_data.processed.xy
        else:
            x_train, y_train = train_cf_data.transform(
                x_train,
                y_train,
                contains_labels=data.contains_labels,
            ).xy
        if valid_cf_data is None:
            x_valid = y_valid = None
        else:
            x_valid, y_valid = valid_cf_data.processed.xy
        is_classification = data.is_classification
        if is_classification is None:
            is_classification = train_cf_data.is_clf
        return IMLPreProcessedData(
            x_train,
            y_train,
            x_valid,
            y_valid,
            IMLDataInfo(
                x_train.shape[-1],
                num_classes=train_cf_data.num_classes,
                is_classification=is_classification,
            ),
        )

    def process_batch(self, batch: np_dict_type) -> np_dict_type:
        return batch

    def dumps(self) -> Any:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = os.path.join(tmp_dir, self.tmp_cf_data_name)
            self.cf_data.save(tmp_name, retain_data=False)
            zip_file = f"{tmp_name}.zip"
            with open(zip_file, "rb") as f:
                cf_data_bytes = f.read()
            os.remove(zip_file)
        return cf_data_bytes

    def loads(self, dumped: Any) -> None:
        if TabularData is None:
            msg = "`carefree-data` needs to be installed to load `MLCarefreeData`"
            raise ValueError(msg)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_name = os.path.join(tmp_dir, self.tmp_cf_data_name)
            zip_file = f"{tmp_name}.zip"
            with open(zip_file, "wb") as f:
                f.write(dumped)
            self.cf_data = TabularData.load(tmp_name)
            os.remove(zip_file)


@DLDataModule.register("ml.carefree")
class MLCarefreeData(IMLData, metaclass=ConfigMeta):
    processor_type = "_internal.carefree"

    processor: _InternalCarefreeMLDataProcessor

    def __init__(
        self,
        x_train: data_type,
        y_train: data_type = None,
        x_valid: data_type = None,
        y_valid: data_type = None,
        *,
        # processor
        processor: Optional[_InternalCarefreeMLDataProcessor] = None,
        data_config: Optional[Dict[str, Any]] = None,
        read_config: Optional[Dict[str, Any]] = None,
        # auxiliary data
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        # common
        num_history: int = 1,
        num_classes: Optional[int] = None,
        is_classification: Optional[bool] = None,
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
        super().__init__(
            x_train,
            y_train,
            x_valid,
            y_valid,
            processor=processor,
            train_others=train_others,
            valid_others=valid_others,
            num_history=num_history,
            num_classes=num_classes,
            is_classification=is_classification,
            shuffle_train=shuffle_train,
            shuffle_valid=shuffle_valid,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            for_inference=for_inference,
        )
        self.data_config = data_config or {}
        self.read_config = read_config or {}
        self.valid_split = valid_split
        self.min_valid_split = min_valid_split
        self.max_valid_split = max_valid_split
        self.max_valid_split_ratio = max_valid_split_ratio
        self.valid_split_order = valid_split_order
        self.contains_labels = contains_labels

    @classmethod
    def make_with(
        cls,
        *args: Any,
        is_classification: Optional[bool] = None,
        cf_data_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MLCarefreeData":
        if TabularData is None:
            msg = "`carefree-data` needs to be installed for making `MLCarefreeData`"
            raise ValueError(msg)
        if cf_data_config is None:
            cf_data_config = {}
        cf_data_config["default_categorical_process"] = "identical"
        if is_classification is not None:
            cf_data_config["task_type"] = "clf" if is_classification else "reg"
        kwargs["is_classification"] = is_classification
        kwargs["data_config"] = cf_data_config
        return cls(*args, **kwargs)


__all__ = [
    "get_weighted_indices",
    "register_ml_data_processor",
    "IMLDataInfo",
    "IMLDataProcessor",
    "IMLPreProcessedData",
    "IMLDataset",
    "IMLLoader",
    "IMLData",
    "MLDataset",
    "MLLoader",
    "MLData",
    "MLInferenceData",
    "MLCarefreeData",
]
