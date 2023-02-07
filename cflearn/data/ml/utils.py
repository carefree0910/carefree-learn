import numpy as np

from enum import Enum
from typing import List
from typing import Tuple
from typing import Union
from typing import NamedTuple
from cftool.misc import print_warning
from cftool.array import is_float
from cftool.array import get_unique_indices


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
    ) -> None:
        if shuffle and order != DataOrder.NONE:
            raise ValueError(
                "`order` should be set to 'none' "
                f"when `shuffle` is set to 'True', but '{order}' found"
            )
        self._order = order
        self._shuffle = shuffle

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
            label_split_indices = map(np.ascontiguousarray, label_split_indices)
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


__all__ = [
    "DataSplitter",
]
