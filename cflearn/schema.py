import json
import math

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from torch import device
from torch import Tensor
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import NamedTuple
from dataclasses import dataclass
from cftool.misc import safe_execute
from cftool.misc import print_warning
from cftool.misc import context_error_handler
from cftool.misc import PureFromInfoMixin
from cftool.misc import DataClassBase
from cftool.misc import ISerializable
from cftool.misc import ISerializableArrays
from cftool.misc import ISerializableDataClass
from cftool.array import to_numpy
from cftool.array import to_torch
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from cftool.pipeline import IBlock
from cftool.pipeline import IPipeline


# types


data_type = Optional[Union[np.ndarray, str]]
texts_type = Union[str, List[str]]
param_type = Union[Tensor, nn.Parameter]
configs_type = Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]
general_config_type = Optional[Union[str, Dict[str, Any]]]
losses_type = Union[Tensor, tensor_dict_type]
forward_results_type = Union[Tensor, tensor_dict_type]
states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
device_type = Optional[Union[int, str, device]]

TData = TypeVar("TData", bound="IData", covariant=True)
TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]
TDataLoaders = Tuple["IDataLoader", Optional["IDataLoader"]]
TDataBundleItem = Optional[Union[data_type, np_dict_type, tensor_dict_type, Any]]
TDataBlock = TypeVar("TDataBlock", bound="IDataBlock", covariant=True)
TDataProcessor = TypeVar("TDataProcessor", bound="DataProcessor", covariant=True)


# collections


data_dict: Dict[str, Type["IData"]] = {}
data_processor_configs: Dict[str, Type["DataProcessorConfig"]] = {}
data_configs: Dict[str, Type["DataConfig"]] = {}


# dataclasses


@dataclass
class MLEncoderSettings(DataClassBase):
    """
    Encoder settings.

    Properties
    ----------
    dim (int) : number of different values of this categorical column.
    methods (str | List[str]) : encoding methods to use for each categorical column.
        * if List[str] is provided and its length > 1, then multiple encoding methods will be used.
    method_configs (Dict[str, Any]) : (flattened) configs of the corresponding encoding methods.
        * even if multiple methods are used, `method_configs` should still be 'flattened'

    """

    dim: int
    methods: Union[str, List[str]] = "embedding"
    method_configs: Optional[Dict[str, Any]] = None

    @property
    def use_one_hot(self) -> bool:
        if self.methods == "one_hot":
            return True
        if isinstance(self.methods, list) and "one_hot" in self.methods:
            return True
        return False

    @property
    def use_embedding(self) -> bool:
        if self.methods == "embedding":
            return True
        if isinstance(self.methods, list) and "embedding" in self.methods:
            return True
        return False


@dataclass
class MLGlobalEncoderSettings(DataClassBase):
    embedding_dim: Optional[int] = None
    embedding_dropout: Optional[float] = None


# data

"""

Design of the `IData` system:

* `IData` itself only holds minimal configurations, but will hold some data - which are
constructed into a `DataBundle` - temporarily, in case we need to use the data immediately
(e.g. use them for training), or need to serialize them.
* Complicated logics are maintained by `DataProcessor`, which is an `IPipeline` constructed
by a series of `IDataBlock`.
* `DataProcessor` itself has no information except for a global `config`, and logics are held
in each `IDataBlock`.
* An `IDataBlock` need to do four jobs:
  * `transform`: transform a `DataBundle` into a new `DataBundle`.
  * `fit_transform`: collect necessary info and perform `transform`.
  * `postprocess_item` (optional): post process an incoming item.
  >   multiple items will be 'collated' into a batch
  * `recover_labels` (optional): recover labels to their original format.


Typical workflows are:

* Training : raw data -> `fit_transform` -> transformed data
             -> fetch items -> `postprocess_item` -> collate -> processed batch
             -> model -> predictions -> `recover_labels`
* Inference: raw data -> `transform` -> transformed data
             -> fetch items -> `postprocess_item` -> collate -> processed batch
             -> model -> predictions -> `recover_labels`

> When serializing, a property called `bundle` (the `DataBundle`) will be saved, which holds
the 'transformed data'. So after the serialization, we don't need to run `fit_transform`/`transform`
anymore, and can reuse the `bundle` property directly.
> However we can also serialize `IData` without saving `bundle` (which is a better choice when
we only want to serialize it for inference). In this case, we need to run `transform` on new datasets.


The above statements indicate that:
* `transform`/`fit_transform` are at the 'pre-calculation' stage.
* `postprocess_item`/`recover_labels` are at the 'on the fly' stage.


Common use cases are:

* ML datasets: will mostly utilize `transform`/`fit_transform`, because most ML datasets
can be transfered into a numpy-based datasets, which should be calculated beforehand
because directly indexing numpy arrays is very fast while streaming them will be slow.

* CV/NLP datasets: will mostly utilize `postprocess_item`, because most CV/NLP datasets
are very large, which means it is impossible to be pre-calculated because that will
cost too much RAM. Instead, common practice is to 'stream' the datasets, which means many
calculations must be done 'on the fly'.

* `recover_labels` might be used across all kinds of datasets, because labels may always need
to be transformed.

"""


def copy_data(data: TDataBundleItem) -> TDataBundleItem:
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: copy_data(v) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return data.copy()
    if isinstance(data, Tensor):
        return data.clone()
    return data


def check_data_is_info(data: TDataBundleItem) -> bool:
    if (
        data is None
        or isinstance(data, dict)
        or isinstance(data, np.ndarray)
        or isinstance(data, Tensor)
    ):
        return False
    try:
        json.dumps([data])
        return True
    except:
        return False


def norm_sw(sample_weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_weights is None:
        return None
    return sample_weights / sample_weights.sum()


def split_sw(sample_weights: sample_weights_type) -> TSplitSW:
    if sample_weights is None:
        train_weights = valid_weights = None
    else:
        if not isinstance(sample_weights, np.ndarray):
            train_weights, valid_weights = sample_weights
        else:
            train_weights, valid_weights = sample_weights, None
    train_weights, valid_weights = map(norm_sw, [train_weights, valid_weights])
    return train_weights, valid_weights


class IDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> Dict[str, Any]:
        pass


class IDataLoader(ABC):
    dataset: IDataset
    batch_size: int

    def __init__(self, *, sample_weights: Optional[np.ndarray] = None):
        self.sample_weights = sample_weights

    @abstractmethod
    def __iter__(self) -> "IDataLoader":
        pass

    @abstractmethod
    def __next__(self) -> np_dict_type:
        pass

    @abstractmethod
    def disable_shuffle(self) -> None:
        pass

    @abstractmethod
    def recover_shuffle(self) -> None:
        pass

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def copy(self) -> "IDataLoader":
        return deepcopy(self)

    def temporarily_disable_shuffle(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, loader: IDataLoader):
                self.loader = loader

            def __enter__(self) -> None:
                self.loader.disable_shuffle()

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.loader.recover_shuffle()

        return _(self)

    def get_full_batch(self) -> np_dict_type:
        batch_size = self.batch_size
        self.batch_size = len(self.dataset)
        full_batch = next(iter(self))
        self.batch_size = batch_size
        return full_batch


class DataArgs(NamedTuple):
    x: TDataBundleItem
    y: TDataBundleItem
    others: Optional[np_dict_type]

    @property
    def xy(self) -> Tuple[TDataBundleItem, TDataBundleItem]:
        return self.x, self.y


@dataclass
class DataBundle(DataClassBase):
    x_train: TDataBundleItem
    y_train: TDataBundleItem = None
    x_valid: TDataBundleItem = None
    y_valid: TDataBundleItem = None
    train_others: Optional[np_dict_type] = None
    valid_others: Optional[np_dict_type] = None

    @property
    def train_args(self) -> DataArgs:
        return DataArgs(self.x_train, self.y_train, self.train_others)

    @property
    def valid_args(self) -> DataArgs:
        return DataArgs(self.x_valid, self.y_valid, self.valid_others)

    def copy(self) -> "DataBundle":
        return DataBundle(*map(copy_data, self.attributes))

    def to_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        for k, v in self.asdict().items():
            if check_data_is_info(v):
                info[k] = v
        return info

    def from_info(self, info: Dict[str, Any]) -> None:
        for k, v in info.items():
            setattr(self, k, v)

    def to_npd(self) -> np_dict_type:
        def _to_np(key: str, data: Union[np.ndarray, Tensor]) -> np.ndarray:
            if isinstance(data, np.ndarray):
                return data
            tensor_keys.append(key)
            return to_numpy(data)

        npd: np_dict_type = {}
        tensor_keys: List[str] = []
        for k, v in self.asdict().items():
            if isinstance(v, dict):
                v = {f"{k}.{vk}": vv for vk, vv in v.items()}
                npd.update({vk: _to_np(vk, vv) for vk, vv in v.items()})
            elif isinstance(v, (np.ndarray, Tensor)):
                npd[k] = _to_np(k, v)
        if tensor_keys:
            npd["__tensor_keys__"] = np.array(tensor_keys)
        return npd

    def from_npd(self, npd: np_dict_type) -> None:
        attr_collections: Dict[str, Union[np_dict_type, tensor_dict_type]] = {}
        tensor_keys = set(npd.pop("__tensor_keys__", np.array([])).tolist())
        for k, v in npd.items():
            attr = None
            if "." in k:
                attr, k = k.split(".", 1)
            if k in tensor_keys:
                v = to_torch(v)
            if attr is None:
                setattr(self, k, v)
            else:
                attr_collections.setdefault(attr, {})[k] = v
        for attr, collection in attr_collections.items():
            setattr(self, attr, collection)

    @classmethod
    def empty(cls) -> "DataBundle":
        return cls(None)


class IDataBlock(PureFromInfoMixin, IBlock, ISerializable, metaclass=ABCMeta):
    """
    `IDataBlock` is a block that can transform data, it's initialization/serialization
    is designed as follows:

    1. The `__init__` method:
    * should not include arguments that do not have default values.
    * should and only should contain arguments which is defined in the `fields` property.

    2. The `fields` property should and only should contain fields which can be initialized
    in the `__init__` method.

    3. The `fit_transform` method should not introduce more fields (except for `INoInitDataBlock`).

    4. `IDataBlock` implements a `to_info` method, which record and only record the properties
    defined in the `fields` property.
    * This method should not be overwritten, except for `INoInitDataBlock`.

    5. `IDataBlock` inherits `PureFromInfoMixin`, which means all properties will be
    properly restored from the info returned by `to_info` method.

    For any class inheriting `IDataBlock`, it can be easily initialized
    with the help of the `get_arguments` function from `cftool.misc`.

    Examples
    --------
    >>> from cftool.misc import get_arguments
    >>>
    >>> class MyBlock(IDataBlock):
    >>>     def __init__(self, foo: int = 1, bar: str = "two"):
    >>>         super().__init__(**get_arguments())
    >>>
    >>>     @property
    >>>     def init_fields(self) -> List[str]:
    >>>         return ["foo", "bar"]
    >>>
    >>>     ...
    >>>
    >>> block = MyBlock()
    >>> print(block.foo, block.bar)  # 1 two
    """

    config: "DataProcessorConfig"
    previous: Dict[str, "IDataBlock"]

    def __init__(self, **kwargs: Any) -> None:
        not_exists_tag = "$$NOT_EXISTS$$"
        for field in self.fields:
            value = kwargs.get(field, not_exists_tag)
            if value == not_exists_tag:
                raise ValueError(
                    f"Argument '{field}' needs to be provided "
                    f"for `{self.__class__.__name__}`."
                )
            setattr(self, field, value)

    # inherit

    def build(self, config: "DataProcessorConfig") -> None:
        self.config = config
        configs = (config.block_configs or {}).setdefault(self.__identifier__, {})
        for field in self.fields:
            setattr(self, field, configs.setdefault(field, getattr(self, field)))

    def to_info(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.fields}

    # abstract

    @property
    @abstractmethod
    def fields(self) -> List[str]:
        pass

    @abstractmethod
    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        """
        This method should not utilize `config`!

        Changes can happen inplace.
        """

    @abstractmethod
    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        """
        This method should prepare necessary info, which might be used
        in the `to_info` method.

        If any necessary info comes from `config`, this method should extract
        them and assign them to the corresponding properties.

        This method will NOT be called in a loading procedure, and the
        necessary info should be loaded in the `from_info` method.

        This method will always assume `for_inference=False`.

        Changes can happen inplace.
        """

    # optional callbacks

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> Any:
        return item

    # changes can happen inplace
    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        return y


class INoInitDataBlock(IDataBlock):
    """
    This type of blocks assume:
    * No property assignments should happen at initialization stage.
    * All properties should be maintained in the `fit_transform` stage.
    """

    @property
    def fields(self) -> List[str]:
        return []


class IRuntimeDataBlock(IDataBlock, metaclass=ABCMeta):
    """
    Runtime blocks will store no information, and will only process the batches
    at runtime. When dealing with CV/NLP datasets, we'll often use this kind of blocks.
    """

    @property
    def fields(self) -> List[str]:
        return []

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        return bundle

    @abstractmethod
    def postprocess_item(self, item: Any) -> Any:
        """changes can happen inplace"""


@dataclass
class DataProcessorConfig(ISerializableDataClass):
    block_names: Optional[List[str]] = None
    block_configs: Optional[Dict[str, Dict[str, Any]]] = None

    @classmethod
    def d(cls) -> Dict[str, Type["DataProcessorConfig"]]:
        return data_processor_configs

    @property
    def default_blocks(self) -> List[IDataBlock]:
        return []

    def add_blocks(self, *blocks: IDataBlock) -> None:
        if self.block_names is None:
            self.block_names = []
        for b in blocks:
            b_id = b.__identifier__
            if b_id in self.block_names:
                print_warning(f"block `{b_id}` already exists, it will be skipped")
            self.block_names.append(b_id)
            if isinstance(b, INoInitDataBlock):
                continue
            if self.block_configs is None:
                self.block_configs = {}
            self.block_configs[b_id] = b.to_info()

    def set_blocks(self, *blocks: IDataBlock) -> None:
        self.block_names = []
        self.add_blocks(*blocks)


@IPipeline.register("base.data_processor")
class DataProcessor(IPipeline):
    config: DataProcessorConfig
    blocks: List[IDataBlock]
    is_ready: bool = False

    # inheritance

    @classmethod
    def init(
        cls: Type[TDataProcessor],
        config: Optional[DataProcessorConfig],
    ) -> TDataProcessor:
        self: DataProcessor = cls()
        self.config = (config or self.config_base()).copy()
        if self.config.block_names is None:
            self.config.set_blocks(*self.config.default_blocks)
        self.before_build_in_init()
        self.build(*(IDataBlock.get(name)() for name in self.config.block_names))  # type: ignore
        return self

    # optional callbacks

    @property
    def config_base(self) -> Type[DataProcessorConfig]:
        return DataProcessorConfig

    @property
    def block_base(self) -> Type[IDataBlock]:
        return IDataBlock

    def before_build_in_init(self) -> None:
        pass

    def after_load(self) -> None:
        self.is_ready = True

    # api

    def _run(self, fn: str, bundle: DataBundle, for_inference: bool) -> DataBundle:
        kw = dict(bundle=bundle.copy(), for_inference=for_inference)
        previous: Dict[str, IDataBlock] = {}
        for block in self.blocks:
            block.previous = previous
            kw["bundle"] = safe_execute(getattr(block, fn), kw)
            previous[block.__identifier__] = block
        return kw["bundle"]  # type: ignore

    def transform(self, bundle: DataBundle, *, for_inference: bool) -> DataBundle:
        return self._run("transform", bundle, for_inference)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        bundle = self._run("fit_transform", bundle, False)
        self.is_ready = True
        return bundle

    # changes can happen inplace
    def postprocess_item(self, item: Any) -> np_dict_type:
        for block in self.blocks:
            item = block.postprocess_item(item)
        return item

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        for block in self.blocks[::-1]:
            y = block.recover_labels(y)
        return y


@dataclass
class DataConfig(ISerializableDataClass):
    for_inference: bool = False
    batch_size: int = 1
    valid_batch_size: Optional[int] = None
    shuffle_train: bool = True
    shuffle_valid: bool = False

    @classmethod
    def d(cls) -> Dict[str, Type["DataConfig"]]:
        return data_configs


class IData(ISerializableArrays, Generic[TData], metaclass=ABCMeta):
    d = data_dict

    train_dataset: IDataset
    valid_dataset: Optional[IDataset]

    train_weights: Optional[np.ndarray]
    valid_weights: Optional[np.ndarray]

    config: DataConfig
    processor: DataProcessor
    bundle: Optional[DataBundle]

    for_inference: bool

    def __init__(self) -> None:
        self.train_weights = None
        self.valid_weights = None

    # abstract

    @abstractmethod
    def get_loaders(self) -> TDataLoaders:
        pass

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        if not self.processor.is_ready:
            raise ValueError(
                "`processor` should be ready before calling `to_info`, "
                "did you forget to call the `fit` method first?"
            )
        return {
            "type": self.__identifier__,
            "processor": self.processor.to_pack().asdict(),
            "config": self.config.to_pack().asdict(),
            "bundle": None if self.bundle is None else self.bundle.to_info(),
        }

    def from_info(self, info: Dict[str, Any]) -> None:
        if self.__identifier__ != info["type"]:
            msg = f"type does not match: {self.__identifier__} != {info['type']}"
            raise ValueError(msg)
        self.processor = self.processor_base.from_pack(info["processor"])
        self.config = self.config_base.from_pack(info["config"])
        bundle_info = info["bundle"]
        if not bundle_info:
            self.bundle = None
        else:
            self.bundle = DataBundle.empty()
            self.bundle.from_info(bundle_info)

    def to_npd(self) -> np_dict_type:
        return {} if self.bundle is None else self.bundle.to_npd()

    def from_npd(self, npd: np_dict_type) -> None:
        if npd:
            if self.bundle is None:
                self.bundle = DataBundle.empty()
            self.bundle.from_npd(npd)

    # optional callback

    @property
    def config_base(self) -> Type[DataConfig]:
        return DataConfig

    @property
    def processor_base(self) -> Type[DataProcessor]:
        return DataProcessor

    def get_bundle(
        self,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> DataBundle:
        args = x_train, y_train, x_valid, y_valid, train_others, valid_others
        return DataBundle(*args)

    def set_sample_weights(self: TData, sample_weights: sample_weights_type) -> TData:
        self.train_weights, self.valid_weights = split_sw(sample_weights)
        return self

    # api

    @classmethod
    def init(
        cls: Type[TData],
        config: Optional[DataConfig] = None,
        processor_config: Optional[DataProcessorConfig] = None,
    ) -> TData:
        self: TData = cls()
        self.bundle = None
        self.config = config or self.config_base()
        self.processor = self.processor_base.init(processor_config)
        return self

    def fit(
        self: TData,
        x_train: data_type,
        y_train: Optional[data_type] = None,
        x_valid: Optional[data_type] = None,
        y_valid: Optional[data_type] = None,
        train_others: Optional[np_dict_type] = None,
        valid_others: Optional[np_dict_type] = None,
        *args: Any,
        **kwargs: Any,
    ) -> TData:
        args = x_train, y_train, x_valid, y_valid, train_others, valid_others, *args
        bundle = self.get_bundle(*args, **kwargs)
        bundle = self.processor.fit_transform(bundle)
        self.bundle = bundle
        return self

    def transform(self, *args: Any, **kwargs: Any) -> DataBundle:
        if not self.processor.is_ready:
            raise ValueError("`processor` should be ready before calling `transform`")
        bundle = self.get_bundle(*args, **kwargs)
        bundle = self.processor.transform(bundle, for_inference=True)
        return bundle

    def recover_labels(self, y: np.ndarray) -> np.ndarray:
        return self.processor.recover_labels(y)


class DataTypes(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"


class ColumnTypes(str, Enum):
    REDUNDANT = "redundant"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


DataProcessorConfig.register("base")(DataProcessorConfig)
DataConfig.register("base")(DataConfig)


__all__ = [
    "data_type",
    "texts_type",
    "param_type",
    "configs_type",
    "general_config_type",
    "losses_type",
    "forward_results_type",
    "states_callback_type",
    "sample_weights_type",
    "MLEncoderSettings",
    "MLGlobalEncoderSettings",
    "norm_sw",
    "split_sw",
    "IDataset",
    "IDataLoader",
    "DataArgs",
    "DataBundle",
    "IDataBlock",
    "INoInitDataBlock",
    "IRuntimeDataBlock",
    "DataProcessorConfig",
    "DataProcessor",
    "DataConfig",
    "IData",
    "DataTypes",
    "ColumnTypes",
]
