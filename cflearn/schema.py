import json
import math
import torch

import numpy as np
import torch.nn as nn

from abc import abstractmethod
from abc import ABC
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from torch import Tensor
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import Protocol
from typing import NamedTuple
from accelerate import Accelerator
from dataclasses import dataclass
from torch.optim import Optimizer
from cftool.misc import filter_kw
from cftool.misc import print_info
from cftool.misc import print_warning
from cftool.misc import safe_execute
from cftool.misc import check_requires
from cftool.misc import shallow_copy_dict
from cftool.misc import get_num_positional_args
from cftool.misc import context_error_handler
from cftool.misc import WithRegister
from cftool.misc import DataClassBase
from cftool.misc import PureFromInfoMixin
from cftool.misc import ISerializable
from cftool.misc import IWithRequirements
from cftool.misc import ISerializableArrays
from cftool.misc import ISerializableDataClass
from cftool.array import to_numpy
from cftool.array import to_torch
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from cftool.pipeline import IBlock
from cftool.pipeline import IPipeline
from torch.optim.lr_scheduler import _LRScheduler

from .types import data_type
from .types import losses_type
from .types import configs_type
from .types import sample_weights_type
from .types import forward_results_type
from .constants import LOSS_KEY
from .constants import INPUT_KEY
from .constants import LABEL_KEY
from .constants import PREDICTIONS_KEY
from .misc.toolkit import is_local_rank_0
from .misc.toolkit import get_clones
from .misc.toolkit import get_device
from .misc.toolkit import get_world_size
from .misc.toolkit import fix_denormal_states
from .misc.toolkit import eval_context
from .misc.toolkit import ONNX

try:
    import onnx
except:
    onnx = None
try:
    from onnxsim import simplify as onnx_simplify

    def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
        initializer_names = [x.name for x in model.graph.initializer]
        return [inp for inp in model.graph.input if inp.name not in initializer_names]

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = [inp.name for inp in get_inputs(model)]
        return input_names

except:
    onnx_simplify = get_input_names = None  # type: ignore


model_dict: Dict[str, Type["IDLModel"]] = {}
monitor_dict: Dict[str, Type["TrainerMonitor"]] = {}
loss_dict: Dict[str, Type["ILoss"]] = {}
metric_dict: Dict[str, Type["IMetric"]] = {}
callback_dict: Dict[str, Type["TrainerCallback"]] = {}
data_dict: Dict[str, Type["IData"]] = {}
data_processor_configs: Dict[str, Type["DataProcessorConfig"]] = {}
data_configs: Dict[str, Type["DataConfig"]] = {}
trainer_configs: Dict[str, Type["TrainerConfig"]] = {}

TData = TypeVar("TData", bound="IData", covariant=True)
TLoss = TypeVar("TLoss", bound="ILoss", covariant=True)
TSplitSW = Tuple[Optional[np.ndarray], Optional[np.ndarray]]
TDataLoaders = Tuple["IDataLoader", Optional["IDataLoader"]]
TDataBundleItem = Optional[Union[data_type, np_dict_type, tensor_dict_type, Any]]
TDataBlock = TypeVar("TDataBlock", bound="IDataBlock", covariant=True)
TDataProcessor = TypeVar("TDataProcessor", bound="DataProcessor", covariant=True)


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


def copy_data(data: TDataBundleItem) -> data_type:
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

    # api

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()


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


# general model


def _forward(
    m: nn.Module,
    batch_idx: int,
    batch: tensor_dict_type,
    general_input_key: str,
    state: Optional["TrainerState"] = None,
    *,
    general_output_key: str = PREDICTIONS_KEY,
    **kwargs: Any,
) -> tensor_dict_type:
    fn = m.forward
    if check_requires(fn, "general_output_key"):
        kwargs["general_output_key"] = general_output_key
    kw = filter_kw(fn, kwargs)
    args: List[Any] = []
    if check_requires(fn, "batch_idx"):
        args.append(batch_idx)
    if get_num_positional_args(fn) > 0:
        args.append(batch if check_requires(fn, "batch") else batch[general_input_key])
    if check_requires(fn, "state"):
        args.append(state)
    rs = m(*args, **kw)
    if not isinstance(rs, dict):
        rs = {general_output_key: rs}
    return rs


class IDLModel(
    nn.Module,
    WithRegister["IDLModel"],
    IWithRequirements,
    metaclass=ABCMeta,
):
    d = model_dict

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    # optional callbacks

    def init_with_trainer(self, trainer: "ITrainer") -> None:
        pass

    def permute_trainer_config(self, trainer_config: "TrainerConfig") -> None:
        pass

    def get_forward_args(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        return (batch[INPUT_KEY],)

    def postprocess(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        forward_results: forward_results_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        if isinstance(forward_results, dict):
            return forward_results
        if isinstance(forward_results, Tensor):
            return {PREDICTIONS_KEY: forward_results}
        raise ValueError(f"unrecognized forward results occurred: {forward_results}")

    # api

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        args = self.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = self.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs

    def onnx_forward(self, batch: tensor_dict_type) -> Any:
        return self.run(0, batch)

    def summary_forward(self, batch: tensor_dict_type) -> None:
        self.onnx_forward(batch)

    def to_onnx(
        self,
        export_file: str,
        input_sample: tensor_dict_type,
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        opset: int = 11,
        simplify: bool = True,
        forward_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        output_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "IDLModel":
        # prepare
        device = get_device(self)
        model = self.cpu()
        if num_samples is not None:
            input_sample = {k: v[:num_samples] for k, v in input_sample.items()}
        onnx_forward = forward_fn or model.onnx_forward
        input_names = sorted(input_sample.keys())
        if output_names is None:
            if forward_fn is not None:
                msg = "`output_names` should be provided when `forward_fn` is provided"
                raise ValueError(msg)
            with eval_context(model):
                forward_results = onnx_forward(shallow_copy_dict(input_sample))
            if not isinstance(forward_results, dict):
                forward_results = {PREDICTIONS_KEY: forward_results}
            output_names = sorted(forward_results.keys())
        # setup
        kwargs = shallow_copy_dict(kwargs)
        kwargs["input_names"] = input_names
        kwargs["output_names"] = output_names
        kwargs["opset_version"] = opset
        kwargs["export_params"] = True
        kwargs["do_constant_folding"] = True
        if dynamic_axes is None:
            dynamic_axes = {}
        elif isinstance(dynamic_axes, list):
            dynamic_axes = {axis: f"axis.{axis}" for axis in dynamic_axes}
        if num_samples is None:
            dynamic_axes[0] = "batch_size"
        dynamic_axes_settings = {}
        for name in input_names + output_names:
            dynamic_axes_settings[name] = dynamic_axes
        kwargs["dynamic_axes"] = dynamic_axes_settings
        kwargs["verbose"] = verbose
        # export

        class ONNXWrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model

            def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
                rs = onnx_forward(batch)
                if isinstance(rs, Tensor):
                    return {k: rs for k in output_names}  # type: ignore
                return {k: rs[k] for k in output_names}  # type: ignore

        m_onnx = ONNXWrapper()
        original_states = model.state_dict()
        fixed_states = fix_denormal_states(original_states, verbose=verbose)
        with eval_context(m_onnx):
            model.load_state_dict(fixed_states)
            torch.onnx.export(
                m_onnx,
                ({k: input_sample[k] for k in input_names}, {}),
                export_file,
                **shallow_copy_dict(kwargs),
            )
            model.load_state_dict(original_states)
            if not simplify:
                return self.to(device)
            if onnx is None:
                print_warning(
                    "`onnx` is not installed, "
                    "so the exported onnx model will not be simplified"
                )
                return self.to(device)
            if onnx_simplify is None or get_input_names is None:
                print_warning(
                    "`onnx-simplifier` is not installed, "
                    "so the exported onnx model will not be simplified"
                )
                return self.to(device)
            try:
                onnx_model = onnx.load(export_file)
                final_input_names = get_input_names(onnx_model)
                model_simplified, check = onnx_simplify(
                    onnx_model,
                    test_input_shapes={
                        name: tensor.shape
                        for name, tensor in input_sample.items()
                        if name in final_input_names
                    },
                )
            except Exception as err:
                if verbose:
                    print_warning(f"Failed to simplify ONNX model ({err})")
                model_simplified = None
                check = False
            if verbose:
                tag = " " if check else " not "
                print_info(f"Simplified ONNX model is{tag}validated!")
            if check and model_simplified is not None:
                onnx.save(model_simplified, export_file)
        return self.to(device)


class IMLModel(IDLModel, metaclass=ABCMeta):
    def __init__(
        self,
        *args: Any,
        encoder_settings: Optional[Dict[str, "MLEncoderSettings"]] = None,
        **kwargs: Any,
    ):
        super().__init__()


class EnsembleFn(Protocol):
    def __call__(self, key: str, tensors: List[Tensor]) -> Tensor:
        pass


class DLEnsembleModel(nn.Module):
    ensemble_fn: Optional[EnsembleFn]

    def __init__(self, m: IDLModel, num_repeat: int) -> None:
        super().__init__()
        self.ms = get_clones(m, num_repeat)
        self.ensemble_fn = None

    def forward(self, *args: Any) -> forward_results_type:
        outputs: Dict[str, List[Tensor]] = {}
        for m in self.ms:
            m_outputs = m(*args)
            if isinstance(m_outputs, Tensor):
                m_outputs = {PREDICTIONS_KEY: m_outputs}
            for k, v in m_outputs.items():
                outputs.setdefault(k, []).append(v)
        final_results: tensor_dict_type = {}
        for k in sorted(outputs):
            if self.ensemble_fn is None:
                v = torch.stack(outputs[k]).mean(0)
            else:
                v = safe_execute(self.ensemble_fn, dict(key=k, tensors=outputs[k]))
            final_results[k] = v
        return final_results

    def run(
        self,
        batch_idx: int,
        batch: tensor_dict_type,
        state: Optional["TrainerState"] = None,
        **kwargs: Any,
    ) -> tensor_dict_type:
        m: IDLModel = self.ms[0]
        args = m.get_forward_args(batch_idx, batch, state, **kwargs)
        forward_results = self(*args)
        outputs = m.postprocess(batch_idx, batch, forward_results, state, **kwargs)
        return outputs


class StepOutputs(NamedTuple):
    forward_results: tensor_dict_type
    loss_dict: Dict[str, float]


class MetricsOutputs(NamedTuple):
    final_score: float
    metric_values: Dict[str, float]
    is_positive: Dict[str, bool]


class InferenceOutputs(NamedTuple):
    forward_results: np_dict_type
    labels: Optional[np.ndarray]
    metric_outputs: Optional[MetricsOutputs]
    loss_items: Optional[Dict[str, float]]


# trainer types


class TrainerState:
    def __init__(
        self,
        loader: IDataLoader,
        *,
        num_epoch: int,
        max_epoch: int,
        fixed_steps: Optional[int] = None,
        extension: int = 5,
        enable_logging: bool = True,
        min_num_sample: int = 3000,
        snapshot_start_step: Optional[int] = None,
        max_snapshot_file: int = 5,
        num_snapshot_per_epoch: int = 2,
        num_step_per_log: int = 350,
        num_step_per_snapshot: Optional[int] = None,
        max_step_per_snapshot: int = 1000,
        min_snapshot_epoch_gap: int = 0,
    ):
        self.step = self.epoch = 0
        self.batch_size = loader.batch_size * get_world_size()
        self.num_step_per_epoch = len(loader)
        self.num_epoch = num_epoch
        self.max_epoch = max_epoch
        self.fixed_steps = fixed_steps
        self.extension = extension
        self.enable_logging = enable_logging
        self.min_num_sample = min_num_sample
        if snapshot_start_step is None:
            snapshot_start_step = math.ceil(min_num_sample / self.batch_size)
        self.snapshot_start_step = snapshot_start_step
        self.max_snapshot_file = max_snapshot_file
        self.num_snapshot_per_epoch = num_snapshot_per_epoch
        self.num_step_per_log = num_step_per_log
        if num_step_per_snapshot is None:
            num_step_per_snapshot = max(1, int(len(loader) / num_snapshot_per_epoch))
            num_step_per_snapshot = min(max_step_per_snapshot, num_step_per_snapshot)
        self.num_step_per_snapshot = num_step_per_snapshot
        self.max_step_per_snapshot = max_step_per_snapshot
        self.min_snapshot_epoch_gap = min_snapshot_epoch_gap
        self._previous_snapshot_epoch = 0

    def set_terminate(self) -> None:
        self.step = self.epoch = -1

    def update_snapshot_epoch(self) -> None:
        self._previous_snapshot_epoch = self.epoch

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "num_epoch": self.num_epoch,
            "max_epoch": self.max_epoch,
            "fixed_steps": self.fixed_steps,
            "extension": self.extension,
            "enable_logging": self.enable_logging,
            "min_num_sample": self.min_num_sample,
            "snapshot_start_step": self.snapshot_start_step,
            "max_snapshot_file": self.max_snapshot_file,
            "num_snapshot_per_epoch": self.num_snapshot_per_epoch,
            "num_step_per_log": self.num_step_per_log,
            "num_step_per_snapshot": self.num_step_per_snapshot,
            "max_step_per_snapshot": self.max_step_per_snapshot,
        }

    @property
    def is_terminate(self) -> bool:
        return self.epoch == -1

    @property
    def should_train(self) -> bool:
        if self.fixed_steps is not None:
            return self.step < self.fixed_steps
        return self.epoch < self.num_epoch

    @property
    def should_terminate(self) -> bool:
        if self.fixed_steps is None:
            return False
        return self.step == self.fixed_steps

    @property
    def should_monitor(self) -> bool:
        return self.step % self.num_step_per_snapshot == 0

    @property
    def should_log_lr(self) -> bool:
        if not self.enable_logging:
            return False
        denominator = min(self.num_step_per_epoch, 10)
        return self.step % denominator == 0

    @property
    def should_log_losses(self) -> bool:
        if not self.enable_logging:
            return False
        patience = max(4, int(round(self.num_step_per_epoch / 50.0)))
        denominator = min(self.num_step_per_epoch, patience)
        return self.step % denominator == 0

    @property
    def should_log_artifacts(self) -> bool:
        return self.should_log_metrics_msg

    @property
    def should_log_metrics_msg(self) -> bool:
        if not self.enable_logging:
            return False
        if self.is_terminate:
            return True
        min_period = math.ceil(self.num_step_per_log / self.num_step_per_snapshot)
        period = max(1, int(min_period)) * self.num_step_per_snapshot
        return self.step % period == 0

    @property
    def can_snapshot(self) -> bool:
        if self.is_terminate:
            return True
        return self.epoch - self._previous_snapshot_epoch >= self.min_snapshot_epoch_gap

    @property
    def should_start_snapshot(self) -> bool:
        return self.step >= self.snapshot_start_step

    @property
    def should_extend_epoch(self) -> bool:
        return self.epoch == self.num_epoch and self.epoch < self.max_epoch

    @property
    def reached_max_epoch(self) -> bool:
        return self.epoch > self.max_epoch

    @property
    def disable_logging(self) -> context_error_handler:
        class _(context_error_handler):
            def __init__(self, state: TrainerState):
                self.state = state
                self.enabled = state.enable_logging

            def __enter__(self) -> None:
                self.state.enable_logging = False

            def _normal_exit(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.state.enable_logging = self.enabled

        return _(self)


class TrainerMonitor(WithRegister["TrainerMonitor"], metaclass=ABCMeta):
    d = monitor_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def snapshot(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def check_terminate(self, new_score: float) -> bool:
        pass

    @abstractmethod
    def punish_extension(self) -> None:
        pass

    def handle_extension(self, state: TrainerState) -> None:
        if state.should_extend_epoch:
            self.punish_extension()
            new_epoch = state.num_epoch + state.extension
            state.num_epoch = min(new_epoch, state.max_epoch)


class MonitorResults(NamedTuple):
    terminate: bool
    save_checkpoint: bool
    metric_outputs: Optional[MetricsOutputs]


# loss


class ILoss(nn.Module, WithRegister[TLoss], metaclass=ABCMeta):
    d = loss_dict
    placeholder_key = "[PLACEHOLDER]"

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def _reduce(self, losses: Tensor) -> Tensor:
        if self.reduction == "none":
            return losses
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        raise NotImplementedError(f"reduction '{self.reduction}' is not implemented")

    # optional callbacks

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results[PREDICTIONS_KEY], batch[LABEL_KEY]

    def postprocess(
        self,
        losses: losses_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        if not isinstance(losses, dict):
            losses = {LOSS_KEY: losses}
        return {k: self._reduce(v) for k, v in losses.items()}

    # api

    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        args = self.get_forward_args(forward_results, batch, state)
        losses = self(*args)
        losses = self.postprocess(losses, batch, state)
        return losses


class _MultiLoss(ILoss, metaclass=ABCMeta):
    def __init__(
        self,
        reduction: str = "mean",
        *,
        loss_names: Optional[List[str]] = None,
        loss_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(reduction)
        name = self.__class__.__name__
        if loss_names is None:
            raise ValueError(f"`loss_names` should be specified for {name}")
        loss_configs = loss_configs or {}
        loss_weights = loss_weights or {}
        self.loss_weights = {k: loss_weights.get(k, 1.0) for k in loss_names}
        self.base_losses = nn.ModuleList(
            [
                ILoss.make(name, loss_configs.get(name, {}), ensure_safe=True)
                for name in loss_names
            ]
        )

    @staticmethod
    def _inject(key: str, base_losses: losses_type, all_losses: losses_type) -> None:
        if isinstance(base_losses, dict):
            base_losses = base_losses[LOSS_KEY]
        all_losses[key] = base_losses

    def _merge(self, losses: tensor_dict_type) -> None:
        merged = 0.0
        for k, v in self.loss_weights.items():
            k_loss = losses.get(k)
            if k_loss is not None:
                merged += k_loss * v
        losses[LOSS_KEY] = merged


@ILoss.register("multi_task")
class MultiTaskLoss(_MultiLoss):
    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        losses: losses_type = {}
        for loss_ins in self.base_losses:
            self._inject(
                loss_ins.__identifier__,
                loss_ins.run(forward_results, batch, state),
                losses,
            )
        self._merge(losses)
        return losses


@ILoss.register("multi_stage")
class MultiStageLoss(_MultiLoss):
    def run(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> tensor_dict_type:
        predictions = forward_results[PREDICTIONS_KEY]
        losses: tensor_dict_type = {}
        for i, pred in enumerate(predictions):
            forward_results[PREDICTIONS_KEY] = pred
            for loss_ins in self.base_losses:
                self._inject(
                    f"{i}_{loss_ins.__identifier__}",
                    loss_ins.run(forward_results, batch, state),
                    losses,
                )
        self._merge(losses)
        return losses


@ILoss.register(ILoss.placeholder_key)
class PlaceholderLoss(ILoss):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        raise ValueError("`forward` should not be called in `PlaceholderLoss`")


# inference


inferences: Dict[str, Type["IInference"]] = {}


class IInference(WithRegister["IInference"], metaclass=ABCMeta):
    d = inferences
    use_grad_in_predict = False

    @abstractmethod
    def get_outputs(
        self,
        loader: IDataLoader,
        *,
        portion: float = 1.0,
        state: Optional[TrainerState] = None,
        metrics: Optional["IMetric"] = None,
        loss: Optional[ILoss] = None,
        return_outputs: bool = True,
        stack_outputs: bool = True,
        use_tqdm: bool = False,
        **kwargs: Any,
    ) -> InferenceOutputs:
        pass


# metrics


class IMetric(WithRegister["IMetric"], metaclass=ABCMeta):
    d = metric_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    # abstract

    @property
    @abstractmethod
    def is_positive(self) -> bool:
        pass

    @abstractmethod
    def forward(self, *args: Any) -> float:
        pass

    # optional callback

    @property
    def requires_all(self) -> bool:
        """
        Specify whether this Metric needs 'all' data.

        Typical metrics often does not need to evaluate itself on the entire dataset,
        but some does need to avoid corner cases. (for instance, the AUC metrics may
        fail to evaluate itself on only a batch, because the labels in this batch may
        be all the same, which breaks the calculation of AUC).
        """
        return False

    def get_forward_args(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> Tuple[Any, ...]:
        return np_outputs[PREDICTIONS_KEY], np_batch[LABEL_KEY]

    # api

    def run(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> float:
        args = self.get_forward_args(np_batch, np_outputs, loader)
        return self.forward(*args)

    @classmethod
    def fuse(
        cls,
        names: Union[str, List[str]],
        configs: configs_type = None,
        *,
        metric_weights: Optional[Dict[str, float]] = None,
    ) -> "IMetric":
        metrics = IMetric.make_multiple(names, configs)
        if isinstance(metrics, IMetric):
            return metrics
        return MultipleMetrics(metrics, weights=metric_weights)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        metric = self.run(np_batch, np_outputs, loader)
        score = metric * (1.0 if self.is_positive else -1.0)
        k = self.__identifier__
        return MetricsOutputs(score, {k: metric}, {k: self.is_positive})


class MultipleMetrics(IMetric):
    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    @property
    def requires_all(self) -> bool:
        return any(metric.requires_all for metric in self.metrics)

    def forward(self, *args: Any) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[IMetric],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}
        self.__identifier__ = " | ".join(m.__identifier__ for m in metric_list)

    def evaluate(
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        is_positive: Dict[str, bool] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(np_batch, np_outputs, loader)
            w = self.weights.get(metric.__identifier__, 1.0)
            weights.append(w)
            scores.append(metric_outputs.final_score * w)
            metrics_values.update(metric_outputs.metric_values)
            is_positive.update(metric_outputs.is_positive)
        return MetricsOutputs(sum(scores) / sum(weights), metrics_values, is_positive)


# trainer interface


class OptimizerPack(NamedTuple):
    scope: str
    optimizer_name: str
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None


@dataclass
class TqdmSettings(DataClassBase):
    use_tqdm: bool = False
    use_step_tqdm: bool = False
    use_tqdm_in_validation: bool = False
    in_distributed: bool = False
    position: int = 0
    desc: str = "epoch"


class TrainerCallback(WithRegister["TrainerCallback"]):
    d = callback_dict

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @property
    def is_local_rank_0(self) -> bool:
        return is_local_rank_0()

    def initialize(self) -> None:
        pass

    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def mutate_train_loss_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "ITrainer",
    ) -> None:
        pass

    def before_loop(self, trainer: "ITrainer") -> None:
        pass

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        pass

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        pass

    def log_metrics_msg(
        self,
        metric_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
    ) -> None:
        pass

    def log_artifacts(self, trainer: "ITrainer") -> None:
        pass

    def after_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        pass

    def after_monitor(
        self,
        monitor_results: MonitorResults,
        state: TrainerState,
    ) -> None:
        pass

    def finalize(self, trainer: "ITrainer") -> None:
        pass


class ITrainer(ABC):
    config: "TrainerConfig"

    loss: ILoss
    model: IDLModel
    metrics: Optional[IMetric]
    monitors: List[TrainerMonitor]
    callbacks: List[TrainerCallback]
    optimizers: Dict[str, Optimizer]
    schedulers: Dict[str, Optional[_LRScheduler]]
    model_for_training: nn.Module
    accelerator: Accelerator

    state: TrainerState
    train_loader: IDataLoader
    train_loader_copy: IDataLoader
    valid_loader: Optional[IDataLoader]
    inference: IInference

    tqdm_settings: TqdmSettings

    @property
    @abstractmethod
    def export_config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def use_tqdm_in_validation(self) -> bool:
        pass

    @property
    @abstractmethod
    def validation_loader(self) -> IDataLoader:
        pass

    @property
    @abstractmethod
    def input_sample(self) -> tensor_dict_type:
        pass

    @property
    @abstractmethod
    def workspace(self) -> str:
        pass

    @property
    @abstractmethod
    def checkpoint_folder(self) -> str:
        pass

    @property
    @abstractmethod
    def has_checkpoint_folder(self) -> bool:
        pass

    # init

    @abstractmethod
    def _init_finetune(self) -> None:
        pass

    # core

    @abstractmethod
    def post_loss_step(self, loss_dict: tensor_dict_type) -> None:
        pass

    @abstractmethod
    def clip_norm_step(self) -> None:
        pass

    @abstractmethod
    def optimizer_step(self) -> None:
        pass

    @abstractmethod
    def scheduler_step(self) -> None:
        pass

    @abstractmethod
    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        pass

    @abstractmethod
    def _monitor_step(self, step_outputs: StepOutputs) -> MonitorResults:
        pass

    @abstractmethod
    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        pass

    @abstractmethod
    def fit(
        self,
        data: IData,
        loss: ILoss,
        model: IDLModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        config_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        cuda: Optional[str] = None,
    ) -> "ITrainer":
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        pass


# configs


class PrecisionType(str, Enum):
    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class TrainerConfig(ISerializableDataClass):
    state_config: Optional[Dict[str, Any]] = None
    workspace: str = "_logs"
    create_sub_workspace: bool = True
    num_epoch: int = 40
    max_epoch: int = 1000
    fixed_epoch: Optional[int] = None
    fixed_steps: Optional[int] = None
    log_steps: Optional[int] = None
    valid_portion: float = 1.0
    mixed_precision: Union[str, PrecisionType] = PrecisionType.NO
    clip_norm: float = 0.0
    metric_names: Optional[Union[str, List[str]]] = None
    metric_configs: configs_type = None
    metric_weights: Optional[Dict[str, float]] = None
    use_losses_as_metrics: Optional[bool] = None
    loss_metrics_weights: Optional[Dict[str, float]] = None
    recompute_train_losses_in_eval: bool = True
    monitor_names: Optional[Union[str, List[str]]] = None
    monitor_configs: Optional[Dict[str, Any]] = None
    auto_callback: bool = True
    callback_names: Optional[Union[str, List[str]]] = None
    callback_configs: Optional[Dict[str, Any]] = None
    lr: Optional[float] = None
    optimizer_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    optimizer_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    update_scheduler_per_epoch: bool = False
    optimizer_settings: Optional[Dict[str, Dict[str, Any]]] = None
    use_zero: bool = False
    finetune_config: Optional[Dict[str, Any]] = None
    tqdm_settings: Optional[Dict[str, Any]] = None

    @classmethod
    def d(cls) -> Dict[str, Type["TrainerConfig"]]:
        return trainer_configs


@dataclass
class Config(TrainerConfig):
    loss_name: Optional[str] = None
    loss_config: Optional[Dict[str, Any]] = None
    in_loading: bool = False
    allow_no_loss: bool = False
    cudnn_benchmark: bool = False

    def to_debug(self) -> None:
        self.fixed_steps = 1
        self.valid_portion = 1.0e-4

    @property
    def trainer_config(self) -> TrainerConfig:
        return safe_execute(TrainerConfig, self.asdict())


@dataclass
class _DLConfig:
    model_name: str = ""
    model_config: Optional[Dict[str, Any]] = None
    num_repeat: Optional[int] = None
    inference_type: str = "dl"


@dataclass
@Config.register("dl")
class DLConfig(Config, _DLConfig):
    def sanity_check(self) -> None:
        if not self.model_name:
            raise ValueError("`model_name` should be provided")


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


@dataclass
@Config.register("ml")
class MLConfig(DLConfig):
    """
    * encoder_settings: used by `Encoder`.
    * global_encoder_settings: used by `Encoder`.
    * index_mapping: since there might be some redundant columns, we may need to
    map the original keys of the `encoder_settings` to the new ones.
    * infer_encoder_settings: whether infer the `encoder_settings` based on
    information gathered by `RecognizerBlock`.
    """

    encoder_settings: Optional[Dict[str, MLEncoderSettings]] = None
    global_encoder_settings: Optional[MLGlobalEncoderSettings] = None
    index_mapping: Optional[Dict[str, int]] = None
    infer_encoder_settings: bool = True

    def from_info(self, info: Dict[str, Any]) -> None:
        super().from_info(info)
        if self.encoder_settings is not None:
            self.encoder_settings = {
                str_idx: MLEncoderSettings(**settings)
                for str_idx, settings in self.encoder_settings.items()
            }
        ges = self.global_encoder_settings
        if ges is not None:
            self.global_encoder_settings = MLGlobalEncoderSettings(**ges)


__all__ = [
    "_forward",
    "loss_dict",
    "metric_dict",
    "callback_dict",
    "IData",
    "DataBundle",
    "IDataBlock",
    "IRuntimeDataBlock",
    "DataProcessorConfig",
    "DataProcessor",
    "DataConfig",
    "IDataset",
    "IDataLoader",
    "IDLModel",
    "StepOutputs",
    "TrainerState",
    "TrainerMonitor",
    "MonitorResults",
    "ILoss",
    "MultiTaskLoss",
    "MultiStageLoss",
    "ONNX",
    "InferenceOutputs",
    "IInference",
    "MetricsOutputs",
    "IMetric",
    "MultipleMetrics",
    "OptimizerPack",
    "TqdmSettings",
    "TrainerCallback",
    "ITrainer",
    "PrecisionType",
    "TrainerConfig",
    "Config",
    "DLConfig",
    "MLEncoderSettings",
    "MLGlobalEncoderSettings",
    "MLConfig",
]
