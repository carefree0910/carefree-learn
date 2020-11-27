import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from cftool.ml import ModelPattern
from cftool.misc import LoggingMixin
from cfdata.tabular import DataLoader
from cfdata.tabular import TabularData


class PipelineProtocol(LoggingMixin, metaclass=ABCMeta):
    def __init__(self) -> None:
        self.data = TabularData.simple("reg", simplify=True)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ) -> "PipelineProtocol":
        self.data.read(x, y)
        return self._core(x, y, x_cv, y_cv)

    @abstractmethod
    def _core(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ) -> "PipelineProtocol":
        pass

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    def to_pattern(self, **predict_config: Any) -> ModelPattern:
        return ModelPattern(predict_method=lambda x: self.predict(x, **predict_config))


class DataProtocol(TabularData):
    pass


class DataLoaderProtocol(DataLoader):
    pass


__all__ = [
    "PipelineProtocol",
    "DataProtocol",
    "DataLoaderProtocol",
]
