import numpy as np

from typing import List
from typing import Union
from typing import Optional
from cftool.types import arr_type
from cftool.types import np_dict_type

from .utils import TArrayDict
from .utils import IArrayDataset
from .utils import IArrayDictDataset
from .utils import IArrayDataMixin
from ..schema import IData
from ..schema import DataArgs
from ..schema import DataProcessor
from ..toolkit import tensor_batch_to_np
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import ORIGINAL_LABEL_KEY


class ArrayDataset(IArrayDataset):
    def __init__(
        self,
        x: arr_type,
        y: Optional[arr_type],
        processor: DataProcessor,
        others: Optional[TArrayDict] = None,
        *,
        for_inference: bool = False,
    ):
        self.x = x
        self.y = y
        self.processor = processor
        self.others = others
        self.for_inference = for_inference

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> np_dict_type:
        if not isinstance(item, int) and isinstance(self.x, list):
            batch = {INPUT_KEY: [self.x[i] for i in item]}
        else:
            batch = {INPUT_KEY: self.x[item]}
        if self.y is not None:
            label = self.y[item]
            batch.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        if self.others is not None:
            for k, v in self.others.items():
                batch[k] = v[item]
        batch = self.processor.postprocess_item(batch, for_inference=self.for_inference)
        batch = tensor_batch_to_np(batch)
        return batch


class ArrayDictDataset(IArrayDictDataset):
    def __init__(
        self,
        x: TArrayDict,
        y: Optional[arr_type],
        processor: DataProcessor,
        *,
        for_inference: bool = False,
    ):
        self.x = x
        self.y = y
        self.processor = processor
        self.x_keys = sorted(self.x)
        self.for_inference = for_inference

    def __len__(self) -> int:
        return len(self.x[self.x_keys[0]])

    def __getitem__(self, item: Union[int, List[int], np.ndarray]) -> TArrayDict:
        batch = {k: self.x[k][item] for k in self.x_keys}
        if self.y is not None:
            label = self.y[item]
            batch.update({LABEL_KEY: label, ORIGINAL_LABEL_KEY: label})
        batch = self.processor.postprocess_item(batch, for_inference=self.for_inference)
        return batch


@IData.register("array")
class ArrayData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs, for_inference: bool) -> ArrayDataset:
        return ArrayDataset(
            *data_args.xy,
            self.processor,
            data_args.others,
            for_inference=for_inference,
        )


@IData.register("array_dict")
class ArrayDictData(IArrayDataMixin, IData):  # type: ignore
    def get_dataset(self, data_args: DataArgs, for_inference: bool) -> ArrayDictDataset:
        return ArrayDictDataset(
            *data_args.xy,
            self.processor,
            for_inference=for_inference,
        )


__all__ = [
    "ArrayDataset",
    "ArrayDictDataset",
    "ArrayData",
    "ArrayDictData",
]
