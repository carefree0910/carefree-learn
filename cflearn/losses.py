import torch

import torch.nn as nn

from typing import *
from torch.nn import functional
from abc import ABCMeta, abstractmethod


class LossBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, config: Dict[str, Any], reduction: str = "mean"):
        super().__init__()
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
        predictions: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # return losses without reduction
        pass

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = self._core(predictions, target)
        return self._reduce(losses)


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
        self._alpha = alpha

    def _core(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._input_logits:
            prob_mat = predictions.view(-1, predictions.shape[-1]) + self._eps
        else:
            logits_mat = predictions.view(-1, predictions.shape[-1])
            prob_mat = functional.softmax(logits_mat, dim=1) + self._eps
        target_column = target.view(-1, 1)
        gathered_prob_flat = prob_mat.gather(dim=1, index=target_column).view(-1)
        gathered_log_prob_flat = gathered_prob_flat.log()
        if self._alpha is not None:
            if isinstance(self._alpha, list):
                self._alpha = torch.tensor(self._alpha).to(predictions)
            alpha_target = self._alpha.gather(dim=0, index=target_column.view(-1))
            gathered_log_prob_flat = gathered_log_prob_flat * alpha_target
        return -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma


__all__ = ["FocalLoss"]
