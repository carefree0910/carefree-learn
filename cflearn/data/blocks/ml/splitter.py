import numpy as np

from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import NamedTuple
from dataclasses import dataclass
from cftool.misc import print_warning
from cftool.array import is_float
from cftool.array import get_unique_indices

from .recognizer import RecognizerBlock
from ....schema import DataBundle
from ....schema import ColumnTypes
from ....schema import INoInitDataBlock


num_split_type = Union[int, float]
xy_type = Tuple[np.ndarray, np.ndarray]


def get_num(n: num_split_type, total: int) -> int:
    if n < 1.0 or (n == 1.0 and isinstance(n, float)):
        return int(round(total * n))
    return int(n)


def preprocess_labels(y: np.ndarray) -> np.ndarray:
    if len(y.shape) == 1:
        y = y[..., None]
    return y


class DataOrder(str, Enum):
    NONE = "none"
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"


class DataSplit(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    x_remained: np.ndarray
    y_remained: np.ndarray
    split_indices: np.ndarray
    remained_indices: np.ndarray

    @property
    def xy(self) -> xy_type:
        return self.x, self.y

    @property
    def xy_remained(self) -> xy_type:
        return self.x_remained, self.y_remained


class DataSplitter:
    """
    Util class for dividing dataset based on task type
    * If it's regression task, it's not difficult to split data
    * If it's classification task, we need to split data based on the labels, because we need to:
      * ensure the divided data contain all labels available
      * ensure the label distribution is identical across all divided data.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> from cflearn.data.ml.utils import DataSplitter
    >>>
    >>> x = np.arange(12).reshape([6, 2])
    >>> # create an imbalance dataset
    >>> y = np.zeros(6, int)
    >>> y[[-1, -2]] = 1
    >>> splitter = DataSplitter()
    >>> # labels in result will keep its ratio
    >>> result = splitter.split(x, y, 3)
    >>> print(result.y.ravel())  # [0 0 1]
    >>> result = splitter.split(x, y, 0.5)
    >>> print(result.y.ravel())  # [0 0 1]
    >>> # at least one sample of each class will be kept
    >>> y[-2] = 0
    >>> result = splitter.split(x, y, 2)
    >>> # [0 0 0 0 0 1] [0 1]
    >>> print(y, result.dataset.y.ravel())

    """

    def __init__(
        self,
        *,
        order: DataOrder = DataOrder.NONE,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> None:
        if shuffle and order != DataOrder.NONE:
            raise ValueError(
                "`order` should be set to 'none' "
                f"when `shuffle` is set to 'True', but '{order}' found"
            )
        self._order = order
        self._shuffle = shuffle
        self._verbose = verbose

    # internal

    def _split_with_indices(
        self,
        x: np.ndarray,
        y: np.ndarray,
        split_indices: np.ndarray,
        remained_indices: np.ndarray,
    ) -> DataSplit:
        return DataSplit(
            x[split_indices],
            y[split_indices],
            x[remained_indices],
            y[remained_indices],
            split_indices,
            remained_indices,
        )

    # api

    @property
    def order(self) -> DataOrder:
        return self._order

    @order.setter
    def order(self, value: DataOrder) -> None:
        self._order = value
        if self._shuffle and value != DataOrder.NONE:
            if self._verbose:
                print_warning(
                    f"`order` is set to '{value}', "
                    "so `shuffle` will be set to 'False' by force"
                )
            self._shuffle = False

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool) -> None:
        self._shuffle = value
        if value and self._order != DataOrder.NONE:
            if self._verbose:
                print_warning(
                    f"`shuffle` is set to 'True', "
                    "so `order` will be set to 'none' by force"
                )
            self._order = DataOrder.NONE

    # will treat float `y` as regression datasets and int `y` as classification datasets
    def split(self, x: np.ndarray, y: np.ndarray, n: num_split_type) -> DataSplit:
        method = "reg" if is_float(y) else "clf"
        return getattr(self, f"split_{method}")(x, y, n)

    # split regression datasets
    def split_reg(self, x: np.ndarray, y: np.ndarray, n: num_split_type) -> DataSplit:
        y = preprocess_labels(y)
        total = len(x)
        n = get_num(n, total)
        if self._shuffle:
            base = np.random.permutation(total)
        else:
            base = np.arange(total)
            if self._order == DataOrder.BOTTOM_UP:
                base = np.ascontiguousarray(base[::-1])
        split_indices = base[:n]
        remained_indices = base[n:]
        return self._split_with_indices(x, y, split_indices, remained_indices)

    # split classification datasets
    def split_clf(self, x: np.ndarray, y: np.ndarray, n: num_split_type) -> DataSplit:
        y = preprocess_labels(y)
        total = len(x)
        n = get_num(n, total)
        res = get_unique_indices(y.ravel())
        label_counts = res.unique_cnt
        label_ratios = label_counts.astype(np.float64) / total
        unique_labels = res.unique.astype(int)
        num_unique_labels = len(unique_labels)
        if n < num_unique_labels:
            raise ValueError(
                f"at least {num_unique_labels} samples are required because "
                f"we have {num_unique_labels} unique labels"
            )
        label_split_indices = [indices.astype(int) for indices in res.split_indices]
        if num_unique_labels == 1:
            raise ValueError(
                "only 1 unique label is detected, "
                "which is invalid in classification task"
            )
        if self._shuffle:
            tuple(map(np.random.shuffle, label_split_indices))
        elif self._order == DataOrder.BOTTOM_UP:
            label_split_indices = [indices[::-1] for indices in label_split_indices]
            label_split_indices = list(map(np.ascontiguousarray, label_split_indices))
        rounded = np.round(n * label_ratios).astype(int)
        num_samples_per_label = np.maximum(1, rounded)
        # -num_unique_labels <= num_samples_exceeded <= num_unique_labels
        num_samples_exceeded = num_samples_per_label.sum() - n
        # adjust num_samples_per_label to make sure `n` samples are split out
        if num_samples_exceeded != 0:
            sign = np.sign(num_samples_exceeded)
            num_samples_exceeded = abs(num_samples_exceeded)
            arange = np.arange(num_unique_labels)
            chosen_indices = arange[num_samples_per_label != 1]
            np.random.shuffle(chosen_indices)
            num_chosen_indices = len(chosen_indices)
            num_tile = int(np.ceil(num_samples_exceeded / num_chosen_indices))
            num_proceeded = 0
            for _ in range(num_tile - 1):
                num_samples_per_label[chosen_indices] -= sign
                num_proceeded += num_chosen_indices
            for idx in chosen_indices[: num_samples_exceeded - num_proceeded]:
                num_samples_per_label[idx] -= sign
        assert num_samples_per_label.sum() == n
        num_overlap = 0
        split_indices_list: List[np.ndarray] = []
        remained_indices_list: List[np.ndarray] = []
        for indices, num_sample_per_label in zip(
            label_split_indices,
            num_samples_per_label,
        ):
            num_samples_available = len(indices)
            split_indices_list.append(indices[:num_sample_per_label])
            if num_sample_per_label >= num_samples_available:
                num_overlap += num_sample_per_label
                remained_indices_list.append(indices)
            else:
                remained_indices_list.append(indices[num_sample_per_label:])
        split_indices = np.hstack(split_indices_list)
        remained_indices = np.hstack(remained_indices_list)
        base = np.zeros(total)
        base[split_indices] += 1
        base[remained_indices] += 1
        assert np.sum(base >= 2) <= num_overlap
        return self._split_with_indices(x, y, split_indices, remained_indices)


@dataclass
class MLSplitterConfig:
    num_split: Optional[num_split_type] = None
    min_split: Optional[int] = None
    max_split: int = 10000
    split_order: DataOrder = DataOrder.NONE
    split_shuffle: bool = True
    is_classification: Optional[bool] = None


@INoInitDataBlock.register("ml_splitter")
class SplitterBlock(INoInitDataBlock):
    config: MLSplitterConfig  # type: ignore
    num_split: Optional[num_split_type]
    min_split: Optional[int]
    max_split: int
    order: DataOrder
    shuffle: bool
    is_classification: bool

    # inheritance

    def to_info(self) -> Dict[str, Any]:
        return dict(
            num_split=self.num_split,
            min_split=self.min_split,
            max_split=self.max_split,
            order=self.order,
            shuffle=self.shuffle,
            is_classification=self.is_classification,
        )

    def transform(self, bundle: DataBundle, for_inference: bool) -> DataBundle:
        if for_inference or bundle.y_train is None or bundle.x_valid is not None:
            return bundle
        if not isinstance(bundle.x_train, np.ndarray):
            return bundle

        if self.num_split is None:
            num = 0.1
            default_min_split = 100
        else:
            num = self.num_split
            default_min_split = 0
        total = len(bundle.x_train)
        if isinstance(num, float):
            num = round(total * num)
        if num <= 0:
            return bundle
        min_split = default_min_split if self.min_split is None else self.min_split
        num = min(self.max_split, max(min_split, num))
        if num == min_split and num >= 0.5 * total:
            print_warning(
                f"`min_split` ({min_split}) exceeds "
                f"half of the total number of samples ({total}), "
                "so `SplitterBlock` will take no effects to "
                "ensure we have enought training samples."
            )
            return bundle
        kw = dict(order=self.order, shuffle=self.shuffle, verbose=self.is_local_rank_0)
        splitter = DataSplitter(**kw)  # type: ignore
        fn = splitter.split_clf if self.is_classification else splitter.split_reg
        split = fn(bundle.x_train, bundle.y_train, num)
        return DataBundle(*split.xy_remained, *split.xy)

    def fit_transform(self, bundle: DataBundle) -> DataBundle:
        self.num_split = self.config.num_split
        self.min_split = self.config.min_split
        self.max_split = self.config.max_split
        self.order = self.config.split_order
        self.shuffle = self.config.split_shuffle
        is_classification = self.config.is_classification
        if is_classification is None:
            if self.recognizer is None:
                is_classification = not is_float(bundle.y_train)
            else:
                is_classification = all(
                    t == ColumnTypes.CATEGORICAL
                    for t in self.recognizer.label_types.values()
                )
        self.is_classification = self.config.is_classification = is_classification
        return self.transform(bundle, False)

    # api

    @property
    def recognizer(self) -> Optional[RecognizerBlock]:
        return self.try_get_previous(RecognizerBlock)


__all__ = [
    "DataOrder",
    "DataSplitter",
    "MLSplitterConfig",
    "SplitterBlock",
]
