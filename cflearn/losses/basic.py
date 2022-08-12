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
from ..protocol import MultiLoss
from ..protocol import ILoss
from ..protocol import TrainerState
from ..constants import LOSS_KEY
from ..constants import INPUT_KEY
from ..constants import LABEL_KEY
from ..constants import PREDICTIONS_KEY
from ..misc.internal_.register import register_loss_module


@register_loss_module("iou")
class IOULoss(nn.Module):
    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return 1.0 - iou(logits, labels)


@register_loss_module("bce")
class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        losses = self.bce(predictions, labels.to(predictions.dtype))
        return losses.mean(tuple(range(1, len(losses.shape))))


@register_loss_module("mae")
class MAELoss(nn.Module):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return F.l1_loss(predictions, labels, reduction="none")


@register_loss_module("sigmoid_mae")
class SigmoidMAELoss(nn.Module):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        losses = F.l1_loss(torch.sigmoid(predictions), labels, reduction="none")
        return losses.mean((1, 2, 3))


@register_loss_module("mse")
class MSELoss(nn.Module):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return F.mse_loss(predictions, labels, reduction="none")


@register_loss_module("recon")
class ReconstructionLoss(nn.Module):
    base_loss: ILoss

    def __init__(self, base_loss_name: str, **kwargs: Any):
        super().__init__()
        self.base_loss = ILoss.make(base_loss_name, kwargs)

    def forward(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        batch = shallow_copy_dict(batch)
        batch[LABEL_KEY] = batch[INPUT_KEY]
        return self.base_loss._core(forward_results, batch, state, **kwargs)


@register_loss_module("quantile")
class QuantileLoss(nn.Module):
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


@register_loss_module("corr")
class CorrelationLoss(nn.Module):
    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return -corr(predictions, labels, get_diagonal=True)


@register_loss_module("cross_entropy")
class CrossEntropyLoss(nn.Module):
    @staticmethod
    def _get_stat(predictions: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        log_prob_mat = F.log_softmax(predictions, dim=1)
        nll_losses = -log_prob_mat.gather(dim=1, index=labels)
        return log_prob_mat, nll_losses

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return self._get_stat(predictions, labels)[1]


@register_loss_module("label_smooth_cross_entropy")
class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self._eps = eps

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        log_prob_mat, nll_losses = CrossEntropyLoss._get_stat(predictions, labels)
        smooth_losses = -log_prob_mat.sum(dim=1, keepdim=True)
        eps = self._eps / log_prob_mat.shape[-1]
        return (1.0 - self._eps) * nll_losses + eps * smooth_losses


@register_loss_module("focal")
class FocalLoss(nn.Module):
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


@MultiLoss.record_prefix()
class MultiTaskLoss(MultiLoss):
    prefix = "multi_task"

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        losses: losses_type = {}
        for loss_ins in self.base_losses:
            self._inject(
                loss_ins.__identifier__,
                loss_ins._core(forward_results, batch, state, **kwargs),
                losses,
            )
        losses[LOSS_KEY] = sum(losses.values())
        return losses


@MultiLoss.record_prefix()
class MultiStageLoss(MultiLoss):
    prefix = "multi_stage"

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        losses: losses_type = {}
        for i, pred in enumerate(predictions):
            forward_results[PREDICTIONS_KEY] = pred
            for loss_ins in self.base_losses:
                self._inject(
                    f"{loss_ins.__identifier__}{i}",
                    loss_ins._core(forward_results, batch, state, **kwargs),
                    losses,
                )
        losses[LOSS_KEY] = sum(losses.values())
        return losses


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
    "MultiStageLoss",
]
