import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from sklearn import metrics
from scipy import stats as ss

from ...protocol import MetricsOutputs
from ...protocol import MetricProtocol
from ...protocol import InferenceOutputs
from ...protocol import DataLoaderProtocol
from ...constants import PREDICTIONS_KEY


@MetricProtocol.register("acc")
class Accuracy(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        logits = outputs.forward_results[PREDICTIONS_KEY]
        predictions = logits.argmax(1)
        labels = outputs.labels.reshape(predictions.shape)  # type: ignore
        return (predictions == labels).mean().item()


@MetricProtocol.register("quantile")
class Quantile(MetricProtocol):
    def __init__(self, q: Any):
        super().__init__()
        if not isinstance(q, float):
            q = np.asarray(q, np.float32).reshape([1, -1])
        self.q = q

    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        diff = outputs.labels - outputs.forward_results[PREDICTIONS_KEY]  # type: ignore
        return np.maximum(self.q * diff, (self.q - 1.0) * diff).mean(0).sum().item()


@MetricProtocol.register("f1")
class F1Score(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = outputs.labels.ravel()  # type: ignore
        predictions = outputs.forward_results[PREDICTIONS_KEY].ravel()
        return metrics.f1_score(labels, predictions)


@MetricProtocol.register("r2")
class R2Score(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = outputs.labels.ravel()  # type: ignore
        predictions = outputs.forward_results[PREDICTIONS_KEY].ravel()
        return metrics.r2_score(labels, predictions)


@MetricProtocol.register("auc")
class AUC(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        logits = outputs.forward_results[PREDICTIONS_KEY]
        num_classes = logits.shape[1]
        probabilities = self.softmax(logits)
        labels = outputs.labels.ravel()  # type: ignore
        if num_classes == 2:
            return metrics.roc_auc_score(labels, probabilities[..., 1])
        return metrics.roc_auc_score(labels, probabilities, multi_class="ovr")


@MetricProtocol.register("mae")
class MAE(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        predictions = outputs.forward_results[PREDICTIONS_KEY]
        return np.mean(np.abs(outputs.labels - predictions)).item()  # type: ignore


@MetricProtocol.register("mse")
class MSE(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        predictions = outputs.forward_results[PREDICTIONS_KEY]
        return np.mean(np.square(outputs.labels - predictions)).item()  # type: ignore


@MetricProtocol.register("ber")
class BER(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = outputs.labels.ravel()  # type: ignore
        predictions = outputs.forward_results[PREDICTIONS_KEY].ravel()
        mat = metrics.confusion_matrix(labels, predictions)
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return (0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))).item()


@MetricProtocol.register("corr")
class Correlation(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = outputs.labels.ravel()  # type: ignore
        predictions = outputs.forward_results[PREDICTIONS_KEY].ravel()
        return float(ss.pearsonr(labels, predictions)[0])


class MultipleMetrics(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    def _core(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[MetricProtocol],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}

    def evaluate(
        self,
        outputs: InferenceOutputs,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(outputs, loader)
            w = self.weights.get(metric.__identifier__, 1.0)
            weights.append(w)
            scores.append(metric_outputs.final_score * w)
            metrics_values.update(metric_outputs.metric_values)
        return MetricsOutputs(sum(scores) / sum(weights), metrics_values)


__all__ = [
    "AUC",
    "BER",
    "MAE",
    "MSE",
    "F1Score",
    "R2Score",
    "Accuracy",
    "Quantile",
    "MultipleMetrics",
]
