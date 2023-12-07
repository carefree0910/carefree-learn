import numpy as np

from typing import Any
from typing import Optional
from cftool.misc import print_warning
from cftool.array import iou
from cftool.array import corr
from cftool.array import softmax
from cftool.array import get_full_logits
from cftool.array import get_label_predictions

from .schema import IMetric
from .toolkit import insert_intermediate_dims
from .constants import LABEL_KEY
from .constants import PREDICTIONS_KEY

try:
    from sklearn import metrics
except:
    metrics = None


@IMetric.register("acc")
class Accuracy(IMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ):
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
        self.threshold = threshold

    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        predictions = get_label_predictions(logits, self.threshold)
        return (predictions == labels).mean().item()


@IMetric.register("quantile")
class Quantile(IMetric):
    def __init__(
        self,
        q: Any,
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ):
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
        if not isinstance(q, np.ndarray):
            q = np.asarray(q, np.float32).reshape([1, -1])
        self.q = q
        self.predictions_key = predictions_key

    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        pred_dim = len(predictions.shape)
        diff = insert_intermediate_dims(labels, predictions) - predictions
        q = self.q.reshape((-1,) + (1,) * (pred_dim - 2))
        error = np.maximum(q * diff, (q - 1.0) * diff)
        return error.mean(0).sum().item()


@IMetric.register("f1")
class F1Score(IMetric):
    def __init__(
        self,
        average: str = "macro",
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ):
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
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
    def __init__(
        self,
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ) -> None:
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
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
    def __init__(
        self,
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ) -> None:
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
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
    def __init__(
        self,
        *,
        labels_key: Optional[str] = LABEL_KEY,
        predictions_key: Optional[str] = PREDICTIONS_KEY,
    ) -> None:
        super().__init__(labels_key=labels_key, predictions_key=predictions_key)
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
