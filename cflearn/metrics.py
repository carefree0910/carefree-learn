import numpy as np

from typing import Any
from cftool.misc import print_warning
from cftool.array import iou
from cftool.array import corr
from cftool.array import softmax
from cftool.array import get_full_logits
from cftool.array import get_label_predictions

from .schema import IMetric

try:
    from sklearn import metrics
except:
    metrics = None


@IMetric.register("acc")
class Accuracy(IMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        predictions = get_label_predictions(logits, self.threshold)
        return (predictions == labels).mean().item()


@IMetric.register("quantile")
class Quantile(IMetric):
    def __init__(self, q: Any):
        super().__init__()
        if not isinstance(q, float):
            q = np.asarray(q, np.float32).reshape([1, -1])
        self.q = q

    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        diff = labels - predictions
        return np.maximum(self.q * diff, (self.q - 1.0) * diff).mean(0).sum().item()


@IMetric.register("f1")
class F1Score(IMetric):
    def __init__(self, average: str = "macro"):
        super().__init__()
        self.average = average
        if metrics is None:
            print_warning("`scikit-learn` needs to be installed for `F1Score`")

    @property
    def is_positive(self) -> bool:
        return True

    @property
    def requires_all(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        if metrics is None:
            return 0.0
        logits = get_full_logits(logits)
        num_classes = logits.shape[1]
        classes = logits.argmax(1)
        labels = labels.ravel()
        if num_classes == 2:
            return metrics.f1_score(labels, classes)
        return metrics.f1_score(labels, classes, average=self.average)


@IMetric.register("r2")
class R2Score(IMetric):
    def __init__(self) -> None:
        super().__init__()
        if metrics is None:
            print_warning("`scikit-learn` needs to be installed for `R2Score`")

    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        if metrics is None:
            return 0.0
        return metrics.r2_score(labels, predictions)


@IMetric.register("auc")
class AUC(IMetric):
    def __init__(self) -> None:
        super().__init__()
        if metrics is None:
            print_warning("`scikit-learn` needs to be installed for `AUC`")

    @property
    def is_positive(self) -> bool:
        return True

    @property
    def requires_all(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        if metrics is None:
            return 0.0
        logits = get_full_logits(logits)
        num_classes = logits.shape[1]
        probabilities = softmax(logits)
        labels = labels.ravel()
        if num_classes == 2:
            return metrics.roc_auc_score(labels, probabilities[..., 1])
        return metrics.roc_auc_score(labels, probabilities, multi_class="ovr")


@IMetric.register("mae")
class MAE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return np.mean(np.abs(labels - predictions)).item()


@IMetric.register("mse")
class MSE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return np.mean(np.square(labels - predictions)).item()


@IMetric.register("ber")
class BER(IMetric):
    def __init__(self) -> None:
        super().__init__()
        if metrics is None:
            print_warning("`scikit-learn` needs to be installed for `AUC`")

    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        if metrics is None:
            return 0.0
        mat = metrics.confusion_matrix(labels.ravel(), predictions.ravel())
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return (0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))).item()


@IMetric.register("corr")
class Correlation(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return corr(predictions, labels, get_diagonal=True).mean().item()


@IMetric.register("iou")
class IOU(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return iou(logits, labels).mean().item()


__all__ = [
    "AUC",
    "BER",
    "IOU",
    "MAE",
    "MSE",
    "F1Score",
    "R2Score",
    "Accuracy",
    "Quantile",
    "Correlation",
]
