import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from cftool.misc import shallow_copy_dict
from cftool.array import iou
from cftool.array import corr
from cftool.array import to_torch
from cftool.types import tensor_dict_type

from ..types import losses_type
from ..schema import ILoss
from ..schema import TrainerState
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import PREDICTIONS_KEY


@ILoss.register("iou")
class IOULoss(ILoss):
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return 1.0 - iou(logits, labels)


@ILoss.register("bce")
class BCELoss(ILoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        losses = self.bce(predictions, labels.to(predictions.dtype))
        return losses.mean(tuple(range(1, len(losses.shape))))


@ILoss.register("mae")
class MAELoss(ILoss):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return F.l1_loss(predictions, labels, reduction="none")


@ILoss.register("sigmoid_mae")
class SigmoidMAELoss(ILoss):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        losses = F.l1_loss(torch.sigmoid(predictions), labels, reduction="none")
        return losses.mean((1, 2, 3))


@ILoss.register("mse")
class MSELoss(ILoss):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return F.mse_loss(predictions, labels, reduction="none")


@ILoss.register("recon")
class ReconstructionLoss(ILoss):
    base_loss: ILoss

    def __init__(
        self, reduction: str = "mean", *, base_loss: str = "mae", **kwargs: Any
    ):
        super().__init__(reduction)
        self.base_loss = ILoss.make(base_loss, kwargs)

    def get_forward_args(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> Tuple[Any, ...]:
        return forward_results, batch, state

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
    ) -> losses_type:
        batch = shallow_copy_dict(batch)
        batch[LABEL_KEY] = batch[INPUT_KEY]
        return self.base_loss.run(forward_results, batch, state)


@ILoss.register("quantile")
class QuantileLoss(ILoss):
    def __init__(self, q: Union[float, List, Tuple, np.ndarray]):
        super().__init__()
        if isinstance(q, float):
            self.register_buffer("q", torch.tensor([q], torch.float32))
        else:
            q = np.asarray(q, np.float32).reshape([1, -1])
            self.register_buffer("q", to_torch(q))

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        quantile_error = labels - predictions
        neg_errors = self.q * quantile_error  # type: ignore
        pos_errors = (self.q - 1.0) * quantile_error  # type: ignore
        quantile_losses = torch.max(neg_errors, pos_errors)
        return quantile_losses.mean(1, keepdim=True)


@ILoss.register("corr")
class CorrelationLoss(ILoss):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return -corr(predictions, labels, get_diagonal=True)


@ILoss.register("cross_entropy")
class CrossEntropyLoss(ILoss):
    @staticmethod
    def _get_stat(predictions: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        log_prob_mat = F.log_softmax(predictions, dim=1)
        nll_losses = -log_prob_mat.gather(dim=1, index=labels)
        return log_prob_mat, nll_losses

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return self._get_stat(predictions, labels)[1]


@ILoss.register("label_smooth_cross_entropy")
class LabelSmoothCrossEntropyLoss(ILoss):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self._eps = eps

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        log_prob_mat, nll_losses = CrossEntropyLoss._get_stat(predictions, labels)
        smooth_losses = -log_prob_mat.sum(dim=1, keepdim=True)
        eps = self._eps / log_prob_mat.shape[-1]
        return (1.0 - self._eps) * nll_losses + eps * smooth_losses


@ILoss.register("focal")
class FocalLoss(ILoss):
    def __init__(
        self,
        *,
        input_logits: bool = True,
        eps: float = 1.0e-6,
        gamma: float = 2.0,
        alpha: Optional[Union[int, float, List, Tuple]] = None,
    ):
        super().__init__()
        self._input_logits = input_logits
        self._eps = eps
        self._gamma = gamma
        if isinstance(alpha, (int, float)):
            alpha = [alpha, 1 - alpha]
        elif isinstance(alpha, (list, tuple)):
            alpha = list(alpha)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", to_torch(np.array(alpha, np.float32)))

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        if not self._input_logits:
            prob_mat = predictions.view(-1, predictions.shape[-1]) + self._eps
        else:
            logits_mat = predictions.view(-1, predictions.shape[-1])
            prob_mat = F.softmax(logits_mat, dim=1) + self._eps
        gathered_prob_flat = prob_mat.gather(dim=1, index=labels).view(-1)
        gathered_log_prob_flat = gathered_prob_flat.log()
        if self.alpha is not None:
            alpha_target = self.alpha.gather(dim=0, index=labels.view(-1))
            gathered_log_prob_flat = gathered_log_prob_flat * alpha_target
        loss = -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma
        return loss.view_as(labels)


__all__ = [
    "IOULoss",
    "MAELoss",
    "MSELoss",
    "QuantileLoss",
    "SigmoidMAELoss",
    "CorrelationLoss",
    "CrossEntropyLoss",
    "LabelSmoothCrossEntropyLoss",
    "FocalLoss",
]
