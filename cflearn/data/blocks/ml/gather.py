import numpy as np

from typing import Any
from typing import Dict
from cftool.array import is_float

from .recognizer import RecognizerBlock
from ....schema import DataBundle
from ....schema import ColumnTypes
from ....schema import INoInitDataBlock


@INoInitDataBlock.register("ml_gather")
class GatherBlock(INoInitDataBlock):
    num_features: int
    num_labels: int

    def to_info(self) -> Dict[str, Any]:
        return dict(num_features=self.num_features, num_labels=self.num_labels)

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        return bundle

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        if not isinstance(bundle.x_train, np.ndarray):
            self.num_features = 0
        else:
            self.num_features = bundle.x_train.shape[-1]
        if bundle.y_train is None or not isinstance(bundle.y_train, np.ndarray):
            self.num_labels = 0
            return bundle
        y_dim = bundle.y_train.shape[-1]
        if y_dim > 1:
            self.num_labels = y_dim
        else:
            b_recognizer = self.try_get_previous(RecognizerBlock)
            if b_recognizer is None:
                if is_float(bundle.y_train):
                    self.num_labels = 1
                else:
                    self.num_labels = len(np.unique(bundle.y_train))
            else:
                key, ctype = list(b_recognizer.label_types.items())[0]
                if ctype == ColumnTypes.NUMERICAL:
                    self.num_labels = 1
                else:
                    self.num_labels = b_recognizer.num_unique_labels[key]
        return bundle


__all__ = ["GatherBlock"]
