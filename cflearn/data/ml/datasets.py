import numpy as np

from typing import Tuple

from ..blocks import ColumnTypes

try:
    from sklearn.utils import Bunch
    from sklearn.datasets import load_iris
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import fetch_california_housing
except:
    Bunch = None
    load_iris = load_digits = load_breast_cancer = fetch_california_housing = None


TXYDataset = Tuple[np.ndarray, np.ndarray]


def _from_bunch(bunch: Bunch, ctype: ColumnTypes) -> TXYDataset:
    x = bunch.data.astype(np.float64)
    y_dtype = np.float64 if ctype == ColumnTypes.NUMERICAL else int
    y = bunch.target.reshape([-1, 1]).astype(y_dtype)
    return x, y


def iris_dataset() -> TXYDataset:
    return _from_bunch(load_iris(), ColumnTypes.CATEGORICAL)


def digits_dataset() -> TXYDataset:
    return _from_bunch(load_digits(), ColumnTypes.CATEGORICAL)


def california_dataset() -> TXYDataset:
    return _from_bunch(fetch_california_housing(), ColumnTypes.NUMERICAL)


def breast_dataset() -> TXYDataset:
    return _from_bunch(load_breast_cancer(), ColumnTypes.CATEGORICAL)


__all__ = [
    "iris_dataset",
    "digits_dataset",
    "california_dataset",
    "breast_dataset",
]
