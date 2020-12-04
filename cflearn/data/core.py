import copy
import torch

import numpy as np

from typing import Any
from cftool.misc import update_dict
from cftool.misc import shallow_copy_dict
from cfdata.types import np_int_type
from cfdata.types import np_float_type
from cfdata.tabular import DataLoader
from cfdata.tabular import ImbalancedSampler
from cfdata.tabular import TabularData as TD

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
    def __next__(self) -> Any:
        sample = DataLoader.__next__(self)
        if self.return_indices:
            (x_batch, labels), indices = sample
        else:
            x_batch, labels = sample
            indices = None
        x_batch = x_batch.astype(np_float_type)
        if self.is_onnx:
            if labels is None:
                labels = np.zeros([*x_batch.shape[:-1], 1], np_int_type)
            arrays = [x_batch, labels]
        else:
            x_batch = to_torch(x_batch)
            if labels is not None:
                labels = to_torch(labels)
                if self.data.is_clf:
                    labels = labels.to(torch.long)
            arrays = [x_batch, labels]

        sample = dict(zip(["x_batch", self.labels_key], arrays))
        if not self.return_indices:
            return sample
        assert indices is not None
        return sample, indices

    def copy(self) -> "DataLoader":
        copied_tabular_loader = copy.copy(self)
        copied_loader = super().copy()
        shallow_copied = shallow_copy_dict(copied_loader.__dict__)
        update_dict(shallow_copied, copied_tabular_loader.__dict__)
        return copied_tabular_loader


__all__ = [
    "TabularData",
    "TabularSampler",
    "TabularLoader",
]
