from cftool.types import np_dict_type
from cftool.types import tensor_dict_type

from .utils import IArrayDataset
from .utils import IArrayDictDataset
from .utils import IArrayDataMixin
from ..schema import IData
from ..schema import DataArgs
from ..misc.toolkit import np_batch_to_tensor


# numpy array


class NumpyDataset(IArrayDataset):
    def before_load(self, npd: np_dict_type) -> np_dict_type:
        return npd


class NumpyDictDataset(IArrayDictDataset):
    def before_load(self, npd: np_dict_type) -> np_dict_type:
        return npd


@IData.register("numpy")
class NumpyData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs) -> NumpyDataset:
        return NumpyDataset(*data_args.xy, self.processor, data_args.others)  # type: ignore


@IData.register("numpy_dict")
class NumpyDictData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs) -> NumpyDictDataset:
        return NumpyDictDataset(*data_args.xy, self.processor)  # type: ignore


# pytorch tensor


class TensorDataset(IArrayDataset):
    def before_load(self, npd: np_dict_type) -> tensor_dict_type:
        return np_batch_to_tensor(npd)


class TensorDictDataset(IArrayDictDataset):
    def before_load(self, npd: np_dict_type) -> tensor_dict_type:
        return np_batch_to_tensor(npd)


@IData.register("tensor")
class TensorData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs) -> TensorDataset:
        return TensorDataset(*data_args.xy, self.processor, data_args.others)  # type: ignore


@IData.register("tensor_dict")
class TensorDictData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs) -> TensorDictDataset:
        return TensorDictDataset(*data_args.xy, self.processor)  # type: ignore


__all__ = [
    "NumpyDataset",
    "NumpyDictDataset",
    "NumpyData",
    "NumpyDictData",
    "TensorDataset",
    "TensorDictDataset",
    "TensorData",
    "TensorDictData",
]
