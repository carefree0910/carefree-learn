import torch

import numpy as np

from typing import Union
from typing import Optional
from cfdata.types import np_int_type
from cfdata.types import np_float_type

from ..types import np_dict_type
from ..types import tensor_dict_type
from ..types import tensor_batch_type
from ..api.protocol import DataProtocol
from ..api.protocol import DataLoaderProtocol
from ..misc.toolkit import to_torch


class TabularData(DataProtocol):
    pass


class TabularLoader(DataLoaderProtocol):
    pass


class PrefetchLoader:
    def __init__(
        self,
        loader: DataLoaderProtocol,
        device: Union[str, torch.device],
        *,
        is_onnx: bool = False,
    ):
        self.loader = loader
        self.device = device
        self.is_onnx = is_onnx
        self.data = loader.data
        self.return_indices = loader.return_indices
        self.stream = None if self.is_cpu else torch.cuda.Stream()
        self.next_batch: Union[np_dict_type, tensor_dict_type]
        self.next_batch_indices: Optional[torch.Tensor]
        self.stop_at_next_batch = False
        self.batch_size = loader.batch_size
        self._num_siamese = loader._num_siamese

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> "PrefetchLoader":
        self.stop_at_next_batch = False
        self.loader.__iter__()
        self.preload()
        return self

    def __next__(self) -> tensor_batch_type:
        if self.stop_at_next_batch:
            raise StopIteration
        if not self.is_cpu:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch, batch_indices = self.next_batch, self.next_batch_indices
        self.preload()
        return batch, batch_indices

    def preload(self) -> None:
        try:
            sample = next(self.loader)
        except StopIteration:
            self.stop_at_next_batch = True
            return None
        indices_tensor: Optional[torch.Tensor]
        if not self.return_indices:
            x_batch, y_batch = sample
            indices_tensor = None
        else:
            (x_batch, y_batch), batch_indices = sample
            indices_tensor = to_torch(batch_indices).to(torch.long)
        self.next_batch = self._collate_batch(x_batch, y_batch)

        if self.is_cpu:
            self.next_batch_indices = indices_tensor
            return None

        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: None if v is None else v.to(self.device, non_blocking=True)
                for k, v in self.next_batch.items()
            }
            if indices_tensor is None:
                self.next_batch_indices = None
            else:
                indices_tensor = indices_tensor.to(self.device, non_blocking=True)
                self.next_batch_indices = indices_tensor

    @property
    def is_cpu(self) -> bool:
        if self.is_onnx:
            return True
        if isinstance(self.device, str):
            return self.device == "cpu"
        return self.device.type == "cpu"

    def _collate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Union[np_dict_type, tensor_dict_type]:
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


__all__ = [
    "TabularData",
    "TabularLoader",
    "PrefetchLoader",
]
