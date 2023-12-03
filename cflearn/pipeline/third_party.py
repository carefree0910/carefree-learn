import numpy as np

from abc import abstractmethod
from abc import ABC
from typing import Any

from .schema import IEvaluationPipeline
from .blocks import BuildMetricsBlock
from ..schema import DLConfig
from ..schema import IDataLoader
from ..schema import MetricsOutputs
from ..constants import INPUT_KEY
from ..constants import PREDICTIONS_KEY


class IPredictor(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass


class SKLearnClassifier(IPredictor):
    def __init__(self, m: Any) -> None:
        self.m = m

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.m.predict_log_proba(x)


class GeneralEvaluationPipeline(IEvaluationPipeline):
    def __init__(self, config: DLConfig, predictor: IPredictor) -> None:
        b_metrics = BuildMetricsBlock()
        b_metrics.build(config)
        if b_metrics.metrics is None:
            raise ValueError(
                "`metrics` should not be `None` for `GeneralPredictor`, "
                "you may try specifying `metric_names` in `config`"
            )
        self.m = predictor
        self.metrics = b_metrics.metrics

    def evaluate(self, loader: IDataLoader) -> MetricsOutputs:
        full_batch = loader.get_full_batch()
        predictions = self.m.predict(full_batch[INPUT_KEY])
        return self.metrics.evaluate(full_batch, {PREDICTIONS_KEY: predictions}, loader)


__all__ = [
    "IPredictor",
    "SKLearnClassifier",
    "GeneralEvaluationPipeline",
]
