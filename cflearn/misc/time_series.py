import numpy as np

from typing import *
from cftool.ml import EnsemblePattern

from ..protocol import DataProtocol


class TSLabelCollator:
    custom_methods: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def __init__(
        self,
        data: DataProtocol,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        if config is None:
            config = {}
        self._init_config(config)

    def __call__(self, y_batch: np.ndarray) -> np.ndarray:
        return self.fn(y_batch)

    def _init_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._method = config.setdefault("method", "average")
        if self._method == "average":
            config.setdefault("num_history", 1)

    @property
    def fn(self) -> Callable[[np.ndarray], np.ndarray]:
        custom_method = TSLabelCollator.custom_methods.get(self._method)
        if custom_method is not None:
            return custom_method
        return getattr(self, f"_{self._method}")

    @staticmethod
    def _last(y_batch: np.ndarray) -> np.ndarray:
        return y_batch[..., -1, :]

    def _average(self, y_batch: np.ndarray) -> np.ndarray:
        num_history = self.config["num_history"]
        extracted = y_batch[..., -num_history:, :]
        if self.data.is_reg:
            return extracted.mean(axis=1)
        return EnsemblePattern.vote(extracted.squeeze(2), self.data.num_classes)


__all__ = ["TSLabelCollator"]
