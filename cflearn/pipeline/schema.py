from abc import abstractmethod
from abc import ABC

from ..schema import IDataLoader
from ..schema import MetricsOutputs


class IEvaluationPipeline(ABC):
    @abstractmethod
    def evaluate(self, loader: IDataLoader) -> MetricsOutputs:
        pass


__all__ = [
    "IEvaluationPipeline",
]
