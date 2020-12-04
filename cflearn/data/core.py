import torch

import numpy as np

from typing import Tuple
from typing import Union
from cfdata.types import np_int_type
from cfdata.types import np_float_type
from cfdata.tabular import DataLoader
from cfdata.tabular import ImbalancedSampler
from cfdata.tabular import TabularData as TD

from ..types import np_dict_type
from ..types import tensor_dict_type
from ..protocol import DataProtocol
from ..protocol import SamplerProtocol
from ..protocol import DataLoaderProtocol
from ..misc.toolkit import to_torch


@DataProtocol.register("tabular")
class TabularData(TD, DataProtocol):
    pass


@SamplerProtocol.register("tabular")
class TabularSampler(ImbalancedSampler, SamplerProtocol):
    pass


@DataLoaderProtocol.register("tabular")
class TabularLoader(DataLoader, DataLoaderProtocol):
    def collate_fn(  # type: ignore
        self,
        sample: Tuple[np.ndarray, np.ndarray],
    ) -> Union[np_dict_type, tensor_dict_type]:
        x_batch, y_batch = sample
        x_batch = x_batch.astype(np_float_type)
        if self.is_onnx:
            if y_batch is None:
                y_batch = np.zeros([*x_batch.shape[:-1], 1], np_int_type)
            arrays = [x_batch, y_batch]
        else:
            x_batch = to_torch(x_batch)
            if y_batch is not None:
                y_batch = to_torch(y_batch)
                if self.data.is_clf:
                    y_batch = y_batch.to(torch.long)
            arrays = [x_batch, y_batch]
        return dict(zip(["x_batch", "y_batch"], arrays))

    def copy(self) -> "DataLoader":
        copied_loader = DataLoader.copy(self)
        copied_loader.is_onnx = self.is_onnx
        copied_loader.collate_fn = self.collate_fn
        return copied_loader


__all__ = [
    "TabularData",
    "TabularSampler",
    "TabularLoader",
]
