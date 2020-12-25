import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import Type
from typing import Tuple
from typing import Callable
from cftool.misc import register_core

from .types import losses_type
from .types import tensor_dict_type
from .misc.toolkit import to_torch


loss_dict: Dict[str, Type["LossBase"]] = {}


class LossBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: Dict[str, Any], reduction: str = "mean"):
        super().__init__()
        self.config = config
        self._init_config(config)
        self._reduction = reduction

    def _init_config(self, config: Dict[str, Any]) -> None:
        pass

    def _reduce(self, losses: torch.Tensor) -> torch.Tensor:
        if self._reduction == "none":
            return losses
        if self._reduction == "mean":
            return losses.mean()
        if self._reduction == "sum":
            return losses.sum()
        raise NotImplementedError(f"reduction '{self._reduction}' is not implemented")

    @abstractmethod
    def _core(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        # return losses without reduction
        pass

    def forward(
        self,
        predictions: tensor_dict_type,
        target: torch.Tensor,
    ) -> losses_type:
        losses = self._core(predictions, target)
        if isinstance(losses, torch.Tensor):
            return self._reduce(losses)
        return {k: self._reduce(v) for k, v in losses.items()}

    @classmethod
    def make(
        cls,
        name: str,
        config: Dict[str, Any],
        reduction: str = "mean",
    ) -> "LossBase":
        return loss_dict[name](config, reduction)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global loss_dict
        return register_core(name, loss_dict)


@LossBase.register("mae")
class MAELoss(LossBase):
    def _core(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        return F.l1_loss(forward_results["predictions"], target, reduction="none")


@LossBase.register("mse")
class MSELoss(LossBase):
    def _core(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        return F.mse_loss(forward_results["predictions"], target, reduction="none")


@LossBase.register("quantile")
class Quantile(LossBase):
    def _init_config(self, config: Dict[str, Any]) -> None:
        q = config.get("q")
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
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        quantile_error = target - forward_results["quantiles"]
        neg_errors = self.q * quantile_error  # type: ignore
        pos_errors = (self.q - 1) * quantile_error  # type: ignore
        quantile_losses = torch.max(neg_errors, pos_errors)
        return quantile_losses.mean(1, keepdim=True)


@LossBase.register("cross_entropy")
class CrossEntropy(LossBase):
    @staticmethod
    def _get_stat(
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_mat = predictions.view(-1, predictions.shape[-1])
        log_prob_mat = F.log_softmax(logits_mat, dim=1)
        nll_losses = -log_prob_mat.gather(dim=1, index=target)
        return log_prob_mat, nll_losses

    def _core(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        return self._get_stat(forward_results["predictions"], target)[1]


@LossBase.register("label_smooth_cross_entropy")
class LabelSmoothCrossEntropy(LossBase):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self._eps = config.setdefault("eps", 0.1)

    def _core(
        self,
        forward_results: tensor_dict_type,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results["predictions"]
        log_prob_mat, nll_losses = CrossEntropy._get_stat(predictions, target)
        smooth_losses = -log_prob_mat.sum(dim=1, keepdim=True)
        eps = self._eps / log_prob_mat.shape[-1]
        return (1.0 - self._eps) * nll_losses + eps * smooth_losses


@LossBase.register("focal")
class FocalLoss(LossBase):
    def _init_config(self, config: Dict[str, Any]) -> None:
        self._input_logits = config.setdefault("input_logits", True)
        self._eps = config.setdefault("eps", 1e-6)
        self._gamma = config.setdefault("gamma", 2.0)
        alpha = config.setdefault("alpha", None)
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
        target: torch.Tensor,
        **kwargs: Any,
    ) -> losses_type:
        predictions = forward_results["predictions"]
        if not self._input_logits:
            prob_mat = predictions.view(-1, predictions.shape[-1]) + self._eps
        else:
            logits_mat = predictions.view(-1, predictions.shape[-1])
            prob_mat = F.softmax(logits_mat, dim=1) + self._eps
        target_column = target.view(-1, 1)
        gathered_prob_flat = prob_mat.gather(dim=1, index=target_column).view(-1)
        gathered_log_prob_flat = gathered_prob_flat.log()
        if self.alpha is not None:
            alpha_target = self.alpha.gather(dim=0, index=target_column.view(-1))
            gathered_log_prob_flat = gathered_log_prob_flat * alpha_target
        return -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma


__all__ = ["LossBase"]
