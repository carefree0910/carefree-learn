import numpy as np

from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import print_warning
from cftool.array import iou
from cftool.array import corr
from cftool.array import softmax
from cftool.array import get_full_logits
from cftool.array import get_label_predictions
from cftool.types import np_dict_type

from .register import register_metric
from .register import IMetric
from ...protocol import MetricProtocol
from ...protocol import DataLoaderProtocol
from ...constants import LABEL_KEY
from ...constants import PREDICTIONS_KEY

try:
    from sklearn import metrics
except:
    metrics = None


@register_metric("acc")
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


@register_metric("quantile")
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


@register_metric("f1")
class F1Score(IMetric):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        if metrics is None:
            print_warning("`scikit-learn` needs to be installed for `F1Score`")

    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        if metrics is None:
            return 0.0
        predictions = get_label_predictions(logits, self.threshold)
        return metrics.f1_score(labels.ravel(), predictions.ravel())


@register_metric("r2")
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


@register_metric("auc")
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


@register_metric("mae")
class MAE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return np.mean(np.abs(labels - predictions)).item()


@register_metric("mse")
class MSE(IMetric):
    @property
    def is_positive(self) -> bool:
        return False

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return np.mean(np.square(labels - predictions)).item()


@register_metric("ber")
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


@register_metric("corr")
class Correlation(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return corr(predictions, labels, get_diagonal=True).mean().item()


@register_metric("iou")
class IOU(IMetric):
    @property
    def is_positive(self) -> bool:
        return True

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # type: ignore
        return iou(logits, labels).mean().item()


@register_metric("aux")
class Auxiliary(IMetric):
    def __init__(
        self,
        base: str,
        base_config: Optional[Dict[str, Any]] = None,
        *,
        key: str,
    ):
        super().__init__()
        self.key = key
        self.base = MetricProtocol.make(base, base_config or {})
        self.__identifier__ = f"{base}_{key}"

    @property
    def is_positive(self) -> bool:
        return self.base.is_positive

    def forward(  # type: ignore
        self,
        np_batch: np_dict_type,
        np_outputs: np_dict_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        return self.base._core(
            {LABEL_KEY: np_batch[self.key]},
            {PREDICTIONS_KEY: np_outputs[self.key]},
            loader,
        )


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
