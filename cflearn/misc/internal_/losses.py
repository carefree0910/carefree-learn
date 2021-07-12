import torch

import numpy as np
import torch.nn.functional as F

from typing import Any
from typing import Tuple
from typing import Optional

from ...constants import *
from ...types import losses_type
from ...types import tensor_dict_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...misc.toolkit import to_torch


@LossProtocol.register("mae")
class MAELoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.l1_loss(predictions, labels, reduction="none")


@LossProtocol.register("mse")
class MSELoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return F.mse_loss(predictions, labels, reduction="none")


@LossProtocol.register("quantile")
class QuantileLoss(LossProtocol):
    def _init_config(self) -> None:
        q = self.config.get("q")
        if q is None:
            raise ValueError("'q' should be provided in Quantile loss")
        if isinstance(q, float):
            self.register_buffer("q", torch.tensor([q], torch.float32))
        else:
            q = np.asarray(q, np.float32).reshape([1, -1])
            self.register_buffer("q", to_torch(q))

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        quantile_error = batch[LABEL_KEY] - forward_results[PREDICTIONS_KEY]
        neg_errors = self.q * quantile_error  # type: ignore
        pos_errors = (self.q - 1.0) * quantile_error  # type: ignore
        quantile_losses = torch.max(neg_errors, pos_errors)
        return quantile_losses.mean(1, keepdim=True)


def corr(
    predictions: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    get_diagonal: bool = False,
) -> torch.Tensor:
    w_sum = 0.0 if weights is None else weights.sum().item()
    if weights is None:
        mean = predictions.mean(0, keepdim=True)
    else:
        mean = (predictions * weights).sum(0, keepdim=True) / w_sum
    vp = predictions - mean
    if weights is None:
        vp_norm = torch.norm(vp, 2, dim=0, keepdim=True)
    else:
        vp_norm = (weights * (vp ** 2)).sum(0, keepdim=True).sqrt()
    if predictions is target:
        mat = vp.t().matmul(vp) / (vp_norm * vp_norm.t())
    else:
        if weights is None:
            target_mean = target.mean(0, keepdim=True)
        else:
            target_mean = (target * weights).sum(0, keepdim=True) / w_sum
        vt = (target - target_mean).t()
        if weights is None:
            vt_norm = torch.norm(vt, 2, dim=1, keepdim=True)
        else:
            vt_norm = (weights.t() * (vt ** 2)).sum(1, keepdim=True).sqrt()
        mat = vt.matmul(vp) / (vp_norm * vt_norm)
    if not get_diagonal:
        return mat
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            "`get_diagonal` is set to True but the correlation matrix "
            "is not a squared matrix, which is an invalid condition"
        )
    return mat.diag()


@LossProtocol.register("corr")
class CorrelationLoss(LossProtocol):
    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return -corr(predictions, labels, get_diagonal=True)


@LossProtocol.register("cross_entropy")
class CrossEntropyLoss(LossProtocol):
    @staticmethod
    def _get_stat(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_prob_mat = F.log_softmax(predictions, dim=1)
        nll_losses = -log_prob_mat.gather(dim=1, index=labels)
        return log_prob_mat, nll_losses

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        return self._get_stat(predictions, labels)[1]


@LossProtocol.register("label_smooth_cross_entropy")
class LabelSmoothCrossEntropyLoss(LossProtocol):
    def _init_config(self) -> None:
        self._eps = self.config.setdefault("eps", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
        log_prob_mat, nll_losses = CrossEntropyLoss._get_stat(predictions, labels)
        smooth_losses = -log_prob_mat.sum(dim=1, keepdim=True)
        eps = self._eps / log_prob_mat.shape[-1]
        return (1.0 - self._eps) * nll_losses + eps * smooth_losses


@LossProtocol.register("focal")
class FocalLoss(LossProtocol):
    def _init_config(self) -> None:
        self._input_logits = self.config.setdefault("input_logits", True)
        self._eps = self.config.setdefault("eps", 1e-6)
        self._gamma = self.config.setdefault("gamma", 2.0)
        alpha = self.config.setdefault("alpha", None)
        if isinstance(alpha, (int, float)):
            alpha = [alpha, 1 - alpha]
        elif isinstance(alpha, (list, tuple)):
            alpha = list(alpha)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", to_torch(np.array(alpha, np.float32)))

    def _core(
        self,
        forward_results: tensor_dict_type,
        batch: tensor_dict_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results[PREDICTIONS_KEY]
        labels = batch[LABEL_KEY]
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
        return -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma


__all__ = [
    "MAELoss",
    "MSELoss",
    "QuantileLoss",
    "CorrelationLoss",
    "CrossEntropyLoss",
    "LabelSmoothCrossEntropyLoss",
    "FocalLoss",
]
